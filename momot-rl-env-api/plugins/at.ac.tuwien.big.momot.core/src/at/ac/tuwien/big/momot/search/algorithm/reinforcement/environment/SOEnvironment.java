package at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment;

import at.ac.tuwien.big.moea.problem.solution.variable.IPlaceholderVariable;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IRewardStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.ISOEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.LocalSearchStrategy;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.ToDoubleFunction;

import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.henshin.interpreter.Engine;
import org.moeaframework.core.Solution;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SOEnvironment<S extends Solution> extends AbstractEnvironment<S> implements ISOEnvironment<S> {

   private final IRewardStrategy<S> rewardStrategy;
   private double prevFitness = .0;
   private double prevConstraintViolation = .0;
   private boolean fitnessEverEvaluated = false; // Add this
   private final boolean debug = false;

   ToDoubleFunction<double[]> fitnessAggregator;
   ToDoubleFunction<double[]> constraintAggregator;
   ToDoubleFunction<EObject> fitnessScaler;
   ToDoubleFunction<EObject> constraintScaler;

   private double[] fitnessNormalizers;
   private double[] constraintNormalizers;

   public SOEnvironment(final ToDoubleFunction<double[]> fitnessAggregator,
         final ToDoubleFunction<double[]> constraintAggregator, final ToDoubleFunction<EObject> fitnessScaler,
         final ToDoubleFunction<EObject> constraintScaler, final IRewardStrategy<S> rewardStrategy,
         final IEncodingStrategy<S> encodingStrategy) {
      super(encodingStrategy);
      this.rewardStrategy = rewardStrategy;

      this.fitnessAggregator = fitnessAggregator;
      this.constraintAggregator = constraintAggregator;
      this.fitnessScaler = fitnessScaler;
      this.constraintScaler = constraintScaler;

   }

   public double[] adaptInitialConstraints(final double[] c) {
      if(this.constraintNormalizers == null) {
         return c;
      }
      final double[] constraints = new double[c.length];
      for(int i = 0; i < constraints.length; i++) {
         constraints[i] = c[i];
         if(this.constraintNormalizers != null && i < this.constraintNormalizers.length) {
            constraints[i] *= this.constraintNormalizers[i];
         }
      }

      return constraints;
   }

   public double[] adaptInitialObjectives(final double[] o) {
      if(this.fitnessNormalizers == null) {
         return o;
      }
      final double[] objectives = new double[o.length];
      for(int i = 0; i < objectives.length; i++) {
         objectives[i] = o[i];
         if(this.fitnessNormalizers != null && i < this.fitnessNormalizers.length) {
            objectives[i] *= this.fitnessNormalizers[i];
         }
      }

      return objectives;
   }

   private double calcReward(final S nextState, final boolean updatePrevs) {
      if(USE_ABSOLUTE_COST) {

         double violation = this.determineConstraintCost(nextState);
         final double quality = this.determineFitnessReward(nextState, updatePrevs);

         if(useLagrangian) {
            if(violation == 0) {
               foundFeasibleState = true;
            }
            currentEpisodeViolation += violation;
            violation *= mu;
         }
         // Lagrangian reward
         final double reward = quality - violation;
         if(debug) {
            System.out.printf("quality-cost: %.3f", reward);
         }

         return reward;
      } else {
         final double reward = this.determineRewardByFitnessFunction(nextState, updatePrevs);
         return reward;
      }
   }

   private double calculateConstraintReward(final double constraintVal) {
      // Reward improvement in constraint violation (reduction is positive)

      double constriantRew = -(constraintVal - prevConstraintViolation);
      if(useLagrangian) {
         if(constraintVal == 0) {
            foundFeasibleState = true;
         }
         constriantRew *= mu;
      }
      return constriantRew;
   }

   private double calculateFitnessReward(final double fitnessVal) {
      // Only fitness development, no constraint passing reward needed
      return -(fitnessVal - prevFitness);
   }

   // private double determineReward(final S state, final boolean updatePrevs) {
   // if(this.rewardStrategy != null && this.rewardStrategy.getRewardMap() != null) {
   //
   // return this.determineRewardByTransformations(this.utils.getApplicationStatesDiff(currentState, state))
   // + this.rewardStrategy.determineAdditionalReward(currentState, state);
   // }
   // return this.determineRewardByFitnessFunction(state, updatePrevs);
   // }

   private double determineConstraintCost(final S state) {
      if(this.constraintAggregator == null) {
         return .0;
         // === Constraint Evaluation ===
      }

      final double cost = this.constraintAggregator.applyAsDouble(this.retrieveConstraints(state)) / initialCost;

      if(debug) {
         System.out.println("cost (abs. w/ baseline): " + cost);
      }

      return cost;
      // final double constraintVal = this.constraintAggregator.applyAsDouble(state.getConstraints());
      // final boolean wasFeasible = prevConstraintViolation == 0.0;
      // final boolean isFeasible = constraintVal == 0.0;

      // Only compute constraint reward if at least one state is infeasible
      // if(!wasFeasible || !isFeasible) {
      // totalReward += calculateConstraintReward(constraintVal);
      // }

      /*
       * if(updatePrevs) {
       * prevConstraintViolation = constraintVal;
       * if(debug) {
       * System.out.println("constraintReward: " + totalReward);
       * System.out.println("prevConstraintViolation: " + prevConstraintViolation);
       * }
       * }
       */

   }

   private double determineFitnessReward(final S state, final boolean updatePrevs) {
      double reward = 0;
      final double[] objectives = this.retrieveObjectives(state);

      final Double fitnessValObj = this.fitnessAggregator.applyAsDouble(objectives);

      if(Math.abs(fitnessValObj) != Double.POSITIVE_INFINITY && !Double.isNaN(fitnessValObj)) {
         final double fitnessVal = fitnessValObj;

         if(fitnessEverEvaluated) {
            final double rewImprovement = -fitnessVal;
            if(debug) {
               System.out.printf("fitness-based reward: %.3f", rewImprovement);
            }
            reward += rewImprovement;

         } else if(updatePrevs) {
            fitnessEverEvaluated = true;
         }

         if(updatePrevs) {
            prevFitness = fitnessVal;

         }
      }
      return reward;
   }

   private double determineRewardByFitnessFunction(final S state, final boolean updatePrevs) {
      double totalReward = 0.0;
      // === Constraint Evaluation ===
      if(this.constraintAggregator != null) {

         final double constraintVal = this.constraintAggregator.applyAsDouble(this.retrieveConstraints(state));
         // final double constraintVal = this.constraintAggregator.applyAsDouble(state.getConstraints());

         final boolean wasFeasible = prevConstraintViolation == 0.0;
         final boolean isFeasible = constraintVal == 0.0;
         // Only compute constraint reward if at least one state is infeasible
         if(!wasFeasible || !isFeasible) {
            totalReward += calculateConstraintReward(constraintVal)
                  * (this.constraintScaler != null ? this.getConstraintScaler(state) : 1.);
         }

         if(updatePrevs) {
            prevConstraintViolation = constraintVal;
            if(debug) {
               System.out.println("constraint-based reward: " + totalReward);
               System.out.println("current violation: " + prevConstraintViolation);
            }

         }
      }

      // === Fitness Evaluation ===
      final double[] objectives = this.retrieveObjectives(state);
      // final double[] objectives = state.getObjectives();

      final Double fitnessValObj = this.fitnessAggregator.applyAsDouble(objectives);

      if(Math.abs(fitnessValObj) != Double.POSITIVE_INFINITY && !Double.isNaN(fitnessValObj)) {
         final double fitnessVal = fitnessValObj;

         if(fitnessEverEvaluated) {

            final double rewImprovement = calculateFitnessReward(fitnessVal)
                  * (this.fitnessScaler != null ? this.getFitnessScaler(state) : 1.);
            if(debug) {
               System.out.printf("fitness reward: %.3f", rewImprovement);
            }
            totalReward += rewImprovement;
         } else if(updatePrevs) {
            fitnessEverEvaluated = true;
         }

         if(updatePrevs) {
            prevFitness = fitnessVal;

         }
      }

      return totalReward;
   }

   // private double determineRewardByFitnessFunction(final S state, final boolean updatePrevs) {
   // double totalReward = 0.0;
   // double constraintVal = 0.0;
   // // === Constraint Evaluation ===
   // if(this.constraintAggregator != null) {
   //
   // constraintVal = this.constraintAggregator.applyAsDouble(this.retrieveConstraints(state));
   // // final double constraintVal = this.constraintAggregator.applyAsDouble(state.getConstraints());
   //
   // final boolean wasFeasible = prevConstraintViolation == 0.0;
   // final boolean isFeasible = constraintVal == 0.0;
   //
   // // Only compute constraint reward if at least one state is infeasible
   // if(!wasFeasible || !isFeasible) {
   // totalReward += calculateConstraintReward(constraintVal);
   // }
   //
   // if(updatePrevs) {
   // prevConstraintViolation = constraintVal;
   // if(debug) {
   // System.out.println("constraint-based reward: " + totalReward);
   // System.out.println("current violation: " + prevConstraintViolation);
   // }
   //
   // }
   // }
   //
   // constraintVal = Arrays.stream(state.getConstraints()).sum();
   // if(constraintVal <= .0) {
   // // === Fitness Evaluation ===
   // final double[] objectives = this.retrieveObjectives(state);
   // // final double[] objectives = state.getObjectives();
   //
   // final Double fitnessValObj = this.fitnessAggregator.applyAsDouble(objectives);
   //
   // if(Math.abs(fitnessValObj) != Double.POSITIVE_INFINITY && !Double.isNaN(fitnessValObj)) {
   // final double fitnessVal = fitnessValObj;
   //
   // if(fitnessEverEvaluated) {
   //
   // final double rewImprovement = calculateFitnessReward(fitnessVal);
   // if(debug) {
   // System.out.printf("fitness reward: %.3f", rewImprovement);
   // }
   // totalReward += rewImprovement;
   // } else if(updatePrevs) {
   // fitnessEverEvaluated = true;
   // }
   //
   // if(updatePrevs) {
   // prevFitness = fitnessVal;
   //
   // }
   // }
   // }
   //
   // return totalReward;
   // }

   // private double determineRewardByTransformations(final List<IApplicationState> ruleAssignments) {
   // double reward = 0;
   // for(final IApplicationState actionRule : ruleAssignments) {
   // final String ruleName = actionRule.getUnit().getName();
   // for(final RuleApplication rule : actionRule.getAppliedRules()) {
   // if(rewardStrategy.getRewardMap().containsKey(rule.getRule().getName())) {
   // reward += rewardStrategy.getRewardMap().get(rule.getRule().getName());
   // }
   // }
   // }
   //
   // return reward;
   // }

   private double getConstraintScaler(final S state) {
      final EObject e = ((TransformationSolution) state).getResultGraph().getRoots().get(0);
      return this.constraintScaler.applyAsDouble(e);
   }

   public Engine getEngine() {
      return ((SolutionProvider) this.solutionProvider).getSearchHelper().getEngine();
   }

   private double getFitnessScaler(final S state) {
      final EObject e = ((TransformationSolution) state).getResultGraph().getRoots().get(0);
      return this.fitnessScaler.applyAsDouble(e);
   }

   @Override
   public String getFunctionName() {
      throw new RuntimeException("Not implemented");
   }

   @Override
   public IEncodingStrategy<S> getProblemEncoder() {
      return this.encodingStrategy;
   }

   @Override
   public double getReward(final S state) {
      throw new RuntimeException("Not implemented");
   }

   @Override
   public SOEnvResponse<S> inferStep(final List<IApplicationState> action) {
      final SOEnvResponse<S> response = new SOEnvResponse<>();

      S nextState = null;

      nextState = this.solutionProvider.generateExtendedSolution(this.currentState, action);

      evaluteSolution(nextState);

      response.setDone(determineIsEpisodeDone(nextState));

      if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
         response.setState(nextState);
         currentState = nextState;
      } else {
         System.out.println("Got");
      }

      return response;
   }

   public boolean isPrevStateFeasible() {
      return this.prevConstraintViolation == .0;
   }

   private S localSearchSO(final List<S> solutions) {
      double maxReward = Double.NEGATIVE_INFINITY;
      S nextState = null;
      if(debug) {
         System.out.println("\n----- Sampling -----");
      }
      for(final S solution : solutions) {

         evaluteSolution(solution);

         if(solution.getNumberOfVariables() == this.currentState.getNumberOfVariables()
               || solution.getNumberOfVariables() > 0
                     && solution.getVariable(solution.getNumberOfVariables() - 1) instanceof IPlaceholderVariable) {
            continue;
         }

         final double nextStateReward = calcReward(solution, false);

         if(nextStateReward > maxReward) {
            maxReward = nextStateReward;
            nextState = solution;
         }
      }
      if(debug) {
         System.out.println("\n----- Sampling end -----\n");
      }

      return nextState;
   }

   @Override
   public S reset() {

      currentState = super.reset();
      foundFeasibleState = false;

      final double initialFitness = this.fitnessAggregator
            .applyAsDouble(this.adaptInitialObjectives(currentState.getObjectives()));

      // final double initialFitness = (Double) fitnessComparator.getValue(currentState);

      if(!Double.isNaN(initialFitness) && Math.abs(initialFitness) != Double.POSITIVE_INFINITY) {
         fitnessEverEvaluated = true;
         prevFitness = initialFitness;
      } else {
         fitnessEverEvaluated = false;
         prevFitness = 0;
      }

      if(this.constraintAggregator != null) {
         prevConstraintViolation = this.constraintAggregator
               .applyAsDouble(this.adaptInitialConstraints(currentState.getConstraints()));
         this.initialCost = prevConstraintViolation;

         if(Double.isNaN(prevConstraintViolation) || Math.abs(prevConstraintViolation) == Double.POSITIVE_INFINITY) {
            throw new RuntimeException("Cannot evalute constraint violations: Turns out as NaN/Infinity value");
         }
      }

      return currentState;
   }

   private double[] retrieveConstraints(final S state) {
      final double[] constraints = new double[state.getNumberOfConstraints()];
      for(int i = 0; i < constraints.length; i++) {
         constraints[i] = (double) state.getAttribute("c_dim_" + String.valueOf(i));
         if(this.constraintNormalizers != null && i < this.constraintNormalizers.length) {
            constraints[i] *= this.constraintNormalizers[i];
         }
      }

      return constraints;
   }

   public double[] retrieveObjectives(final S state) {
      final double[] objectives = new double[state.getNumberOfObjectives()];
      for(int i = 0; i < objectives.length; i++) {
         objectives[i] = (double) state.getAttribute("f_dim_" + String.valueOf(i));
         if(this.fitnessNormalizers != null && i < this.fitnessNormalizers.length) {
            objectives[i] *= this.fitnessNormalizers[i];
         }
      }

      return objectives;
   }

   public void setConstraintNormalizers(final double[] constraintNormalizers) {
      this.constraintNormalizers = constraintNormalizers;
   }

   public void setFitnessNormalizers(final double[] fitnessNormalizers) {
      this.fitnessNormalizers = fitnessNormalizers;
   }

   // @Override
   // public SOEnvResponse<S> step(final LocalSearchStrategy strategy, final List<IApplicationState> action,
   // final int explorationSteps) {
   // final SOEnvResponse<S> response = new SOEnvResponse<>();
   //
   // S nextState = null;
   // int performedExplorationSteps = 0;
   //
   // if(action == null) {
   //
   // switch(strategy) {
   // case GREEDY:
   // final List<S> solutions = new ArrayList<>();
   //
   // for(final S solution : this.solutionProvider.generateNeighbors(currentState, explorationSteps,
   // this.encodingStrategy)) {
   // solutions.add(solution);
   // }
   // performedExplorationSteps = solutions.size();
   // nextState = this.localSearchSO(solutions);
   // break;
   // case NONE:
   // for(final S solution : this.solutionProvider.generateNeighbors(currentState, 1, this.encodingStrategy)) {
   // nextState = solution;
   // }
   // evaluteSolution(nextState);
   // performedExplorationSteps = 1;
   // break;
   // }
   //
   // } else {
   // nextState = this.solutionProvider.generateExtendedSolution(this.currentState, action);
   // evaluteSolution(nextState);
   // }
   //
   // response.setDone(determineIsEpisodeDone(nextState));
   // response.setNumExplorationSteps(performedExplorationSteps);
   //
   // if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
   // response.setReward(determineReward(nextState, true));
   // response.setState(nextState);
   // currentState = nextState;
   //
   // }
   // return response;
   // }

   // public SOEnvResponse<S> step(final String rule, final Map<String, Object> paramValues) {
   // final SOEnvResponse<S> response = new SOEnvResponse<>();
   //
   // final S nextState = this.solutionProvider.generateExtendedSolution(currentState, rule, paramValues);
   //
   // if(nextState == null) {
   // response.setState(null);
   // return response;
   // }
   // response.setState(nextState);
   //
   // function.evaluate((TransformationSolution) nextState);
   //
   //
   //
   // final DoneStatus doneStatus = determineIsEpisodeDone(nextState);
   //
   // final double curFitness = determineReward(nextState, true);
   //
   // response.setReward(curFitness);
   // response.setDone(doneStatus);
   //
   // currentState = nextState;
   //
   // return response;
   // }

   @Override
   public SOEnvResponse<S> step(final LocalSearchStrategy strategy, final List<IApplicationState> action,
         final int explorationSteps) {
      final SOEnvResponse<S> response = new SOEnvResponse<>();

      S nextState = null;
      int performedExplorationSteps = 0;

      if(action == null) {

         switch(strategy) {
            case GREEDY:
               final List<S> solutions = new ArrayList<>();

               for(final S solution : this.solutionProvider.generateNeighbors(currentState, explorationSteps,
                     this.encodingStrategy)) {
                  solutions.add(solution);
               }
               performedExplorationSteps = solutions.size();
               nextState = this.localSearchSO(solutions);
               break;
            case NONE:
               for(final S solution : this.solutionProvider.generateNeighbors(currentState, 1, this.encodingStrategy)) {
                  nextState = solution;
               }
               evaluteSolution(nextState);
               performedExplorationSteps = 1;
               break;
         }

      } else {
         nextState = this.solutionProvider.generateExtendedSolution(this.currentState, action);
         evaluteSolution(nextState);
      }

      // Lagrangian reward
      final double reward = calcReward(nextState, true);

      response.setDone(determineIsEpisodeDone(nextState));
      response.setNumExplorationSteps(performedExplorationSteps);

      if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
         response.setReward(reward);
         response.setState(nextState);
         currentState = nextState;

      }
      return response;
   }

   @Override
   public SOEnvResponse<S> step(final S solution, final INDArray actionProbs) {
      throw new RuntimeException("Not implemented");

      // final SOEnvResponse<S> response = new SOEnvResponse<>();
      //
      // final Object[] actionSolution = this.solutionProvider.generateExtendedSolution(this.encodingStrategy, solution,
      // actionProbs);
      // final int action = (int) actionSolution[0];
      // final S nextState = (S) actionSolution[1];
      //
      // response.setState(nextState);
      // response.setAppliedActionId(action);
      //
      // evaluteSolution(nextState);
      //
      // response.setReward(determineReward(nextState, true));
      // response.setDone(determineIsEpisodeDone(nextState));
      //
      // currentState = nextState;

   }

   @Override
   public SOEnvResponse<S> step(final S solution, final int action) {
      throw new RuntimeException("Not implemented");

      // final SOEnvResponse<S> response = new SOEnvResponse<>();
      //
      // final S nextState = this.solutionProvider.generateExtendedSolution(this.encodingStrategy, solution, action);
      //
      // response.setState(nextState);
      // response.setAppliedActionId(action);
      //
      // evaluteSolution(nextState);
      //
      // response.setReward(determineReward(nextState, true));
      // response.setDone(determineIsEpisodeDone(nextState));
      //
      // currentState = nextState;
      //
      // return response;
   }

   public SOEnvResponse<S> step(final String rule, final Map<String, Object> paramValues) {
      final SOEnvResponse<S> response = new SOEnvResponse<>();

      final S nextState = this.solutionProvider.generateExtendedSolution(currentState, rule, paramValues);

      evaluteSolution(nextState);

      final DoneStatus doneStatus = determineIsEpisodeDone(nextState);

      response.setDone(doneStatus);

      // double violation = this.determineConstraintCost(nextState);
      // final double quality = this.determineFitnessReward(nextState, true);
      //
      // if(useLagrangian) {
      // if(violation == 0) {
      // foundFeasibleState = true;
      // }
      // currentEpisodeViolation += violation;
      // violation *= mu;
      // }
      // Lagrangian reward
      final double reward = calcReward(nextState, true);

      response.setReward(reward);
      response.setAggregatedFitness(
            Arrays.stream(retrieveConstraints(nextState)).sum() + Arrays.stream(retrieveObjectives(nextState)).sum());
      if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {

         response.setState(nextState);
         currentState = nextState;

      } else {
         if(debug) {
            System.out.println("Illegal action:");
            // System.out.println("\tViolation = " + violation);
            // System.out.println("\tQuality = " + quality);
            System.out.println("\t  => Reward = " + reward);

         }
      }

      return response;
   }

   // public SOEnvResponse<S> step(final String rule, final Map<String, Object> paramValues) {
   // final SOEnvResponse<S> response = new SOEnvResponse<>();
   //
   // final S nextState = this.solutionProvider.generateExtendedSolution(currentState, rule, paramValues);
   //
   // evaluteSolution(nextState);
   //
   // final DoneStatus doneStatus = determineIsEpisodeDone(nextState);
   //
   // response.setDone(doneStatus);
   //
   // if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
   // response.setReward(determineReward(nextState, true));
   // response.setState(nextState);
   // currentState = nextState;
   //
   // }
   //
   // return response;
   // }

}
