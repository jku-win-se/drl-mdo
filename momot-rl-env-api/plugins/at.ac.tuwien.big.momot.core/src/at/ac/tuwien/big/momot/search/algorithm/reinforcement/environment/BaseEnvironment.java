package at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment;

import at.ac.tuwien.big.moea.problem.solution.variable.IPlaceholderVariable;
import at.ac.tuwien.big.moea.search.algorithm.local.IFitnessComparator;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IRewardStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.ISOEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.LocalSearchStrategy;
import at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.henshin.interpreter.RuleApplication;
import org.moeaframework.core.Solution;
import org.nd4j.linalg.api.ndarray.INDArray;

public class BaseEnvironment<S extends Solution> extends AbstractEnvironment<S> implements ISOEnvironment<S> {

   protected IFitnessComparator<?, Solution> fitnessComparator;
   protected IFitnessComparator<?, Solution> constraintComparator;

   private final IRewardStrategy<S> rewardStrategy;
   private final String objectiveName;
   private final String constraintName;
   private final IEGraphMultiDimensionalFitnessFunction function;
   private double prevFitness = .0;
   private double prevConstraintViolation = .0;
   private boolean fitnessEverEvaluated = false; // Add this

   public BaseEnvironment(final IEGraphMultiDimensionalFitnessFunction function,
         final IFitnessComparator<?, Solution> fitnessComparator,
         final IFitnessComparator<?, Solution> constraintComparator, final String objectiveName,
         final String constraintName, final IRewardStrategy<S> rewardStrategy,
         final IEncodingStrategy<S> encodingStrategy) {
      super(encodingStrategy);
      this.rewardStrategy = rewardStrategy;
      this.fitnessComparator = fitnessComparator;
      this.constraintComparator = constraintComparator;
      this.objectiveName = objectiveName;
      this.constraintName = constraintName;
      this.function = function;

   }

   // private double calculateConstraintReward(final double constraintVal) {
   //
   // return -(constraintVal - prevConstraintViolation);
   // }
   //
   // private double calculateFitnessReward(final double fitnessVal, final double constraintPassingReward) {
   //
   // final double baseReward = -(fitnessVal - prevFitness);
   //
   // return baseReward + constraintPassingReward;
   // }

   private double calculateConstraintReward(final double constraintVal) {
      // Reward improvement in constraint violation (reduction is positive)
      return -(constraintVal - prevConstraintViolation);
   }

   private double calculateFitnessReward(final double fitnessVal) {
      // Only fitness development, no constraint passing reward needed
      return -(fitnessVal - prevFitness);
   }

   private double determineReward(final S state, final boolean updatePrevs) {
      if(this.rewardStrategy != null && this.rewardStrategy.getRewardMap() != null) {

         return this.determineRewardByTransformations(this.utils.getApplicationStatesDiff(currentState, state))
               + this.rewardStrategy.determineAdditionalReward(currentState, state);
      }
      return this.determineRewardByFitnessFunction(state, updatePrevs);
   }

   private double determineRewardByFitnessFunction(final S state, final boolean updatePrevs) {
      double totalReward = 0.0;

      // === Constraint Evaluation ===
      if(constraintComparator != null) {
         final double constraintVal = (Double) constraintComparator.getValue(state);
         final boolean wasFeasible = prevConstraintViolation == 0.0;
         final boolean isFeasible = constraintVal == 0.0;

         // Only compute constraint reward if at least one state is infeasible
         if(!wasFeasible || !isFeasible) {
            totalReward += calculateConstraintReward(constraintVal);
         }

         if(updatePrevs) {
            prevConstraintViolation = constraintVal;
         }
      }

      // === Fitness Evaluation ===
      final Double fitnessValObj = (Double) fitnessComparator.getValue(state);

      if(fitnessValObj != null && !Double.isNaN(fitnessValObj)) {
         final double fitnessVal = fitnessValObj;

         if(fitnessEverEvaluated) {
            totalReward += calculateFitnessReward(fitnessVal);
         } else {
            fitnessEverEvaluated = true;
         }

         if(updatePrevs) {
            prevFitness = fitnessVal;
         }
      }

      return totalReward;
   }
   //
   // private double determineRewardByFitnessFunction(final S state, final boolean updatePrevs) {
   // double totalReward = 0.0;
   //
   // // === Constraint Evaluation ===
   // if(constraintComparator != null) {
   // final double constraintVal = (Double) constraintComparator.getValue(state);
   // final boolean wasFeasible = prevConstraintViolation == 0.0;
   // final boolean isFeasible = constraintVal == 0.0;
   //
   // // Only compute constraint reward if at least one state is infeasible
   // if(!wasFeasible || !isFeasible) {
   // totalReward += calculateConstraintReward(constraintVal);
   // }
   //
   // // // Check for feasibility transitions
   // // if(wasFeasible && !isFeasible) {
   // // // Breaking feasibility: penalty
   // // totalReward -= 3.0;
   // // } else if(!wasFeasible && isFeasible) {
   // // // Achieving feasibility: bonus
   // // everBeenFeasible = true;
   // // totalReward += 1.0;
   // // }
   //
   // if(updatePrevs) {
   // prevConstraintViolation = constraintVal;
   // }
   // }
   //
   // // === Fitness Evaluation ===
   // final Double fitnessValObj = (Double) fitnessComparator.getValue(state);
   //
   // if(fitnessValObj != null && !Double.isNaN(fitnessValObj)) {
   // final double fitnessVal = fitnessValObj;
   //
   // if(fitnessEverEvaluated) {
   // totalReward += calculateFitnessReward(fitnessVal, 0);
   // } else {
   // fitnessEverEvaluated = true;
   // }
   //
   // if(updatePrevs) {
   // prevFitness = fitnessVal;
   // }
   // }
   //
   // return totalReward;
   // }

   // private double determineRewardByFitnessFunction(final S state, final boolean updatePrevs) {
   // // List<IFitnessDimension> cs = new ArrayList<>(this.function.getConstraints());
   // double constraintPassingReward = .0;
   // if(constraintComparator != null) {
   // final double constraintVal = (Double) constraintComparator.getValue(state);
   // if(constraintVal > 0) {
   // if(everBeenFeasible) {
   // return -5.;
   // }
   // final double rv = calculateConstraintReward(constraintVal);
   //
   // if(updatePrevs) {
   // prevConstraintViolation = constraintVal;
   // }
   // return rv;
   // } else if(prevConstraintViolation > 0) {
   // everBeenFeasible = true;
   // constraintPassingReward = 1.;
   // if(updatePrevs) {
   // prevConstraintViolation = .0;
   // }
   // }
   // }
   //
   // double fitnessVal = (Double) fitnessComparator.getValue(state);
   // fitnessVal = !Double.isNaN(fitnessVal) ? fitnessVal : 0;
   //
   // final double rv = calculateFitnessReward(fitnessVal, constraintPassingReward);
   // if(updatePrevs) {
   // prevFitness = fitnessVal;
   // }
   //
   // return rv;
   //
   // }

   private double determineRewardByTransformations(final List<IApplicationState> ruleAssignments) {
      double reward = 0;
      for(final IApplicationState actionRule : ruleAssignments) {
         for(final RuleApplication rule : actionRule.getAppliedRules()) {
            if(rewardStrategy.getRewardMap().containsKey(rule.getRule().getName())) {
               reward += rewardStrategy.getRewardMap().get(rule.getRule().getName());
            }
         }
      }

      return reward;
   }

   @Override
   public String getFunctionName() {
      return this.objectiveName;
   }

   public FunctionType getFunctionType() {
      return this.function.getObjective(this.getFunctionName()).getFunctionType();
   }

   @Override
   public IEncodingStrategy<S> getProblemEncoder() {
      return this.encodingStrategy;
   }

   @Override
   public double getReward(final S state) {
      return this.determineReward(state, true);
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
      }

      return response;
   }

   private S localSearchSO(final List<S> solutions) {
      double maxReward = Double.NEGATIVE_INFINITY;
      S nextState = null;

      for(final S solution : solutions) {

         evaluteSolution(solution);

         if(solution.getNumberOfVariables() == this.currentState.getNumberOfVariables()
               || solution.getNumberOfVariables() > 0
                     && solution.getVariable(solution.getNumberOfVariables() - 1) instanceof IPlaceholderVariable) {
            continue;
         }

         final double nextStateReward = determineReward(solution, false);

         if(nextStateReward > maxReward) {
            maxReward = nextStateReward;
            nextState = solution;
         }
      }

      return nextState;
   }

   @Override
   public S reset() {
      currentState = super.reset();

      final double initialFitness = (Double) fitnessComparator.getValue(currentState);

      if(!Double.isNaN(initialFitness) && Math.abs(initialFitness) != Double.POSITIVE_INFINITY) {
         fitnessEverEvaluated = true;
         prevFitness = initialFitness;
      } else {
         fitnessEverEvaluated = false;
         prevFitness = 0;
      }
      if(constraintComparator != null) {
         prevConstraintViolation = (Double) constraintComparator.getValue(currentState);
      }

      return currentState;
   }

   @Override
   public void setInitialSolution(final S solution) {
      function.evaluate((TransformationSolution) solution);
      super.setInitialSolution(solution);
      // function.evaluate((TransformationSolution) solution);
      //
      // final double initialFitness = (Double) fitnessComparator.getValue(solution);
      //
      // if(!Double.isNaN(initialFitness) && Math.abs(initialFitness) != Double.POSITIVE_INFINITY) {
      // fitnessEverEvaluated = true;
      // prevFitness = initialFitness;
      // } else {
      // fitnessEverEvaluated = false;
      // prevFitness = 0;
      // }
      //
      // if(constraintComparator != null) {
      // prevConstraintViolation = (Double) constraintComparator.getValue(solution);
      // }

   }

   public SOEnvResponse<S> step(final int action) {
      final SOEnvResponse<S> response = new SOEnvResponse<>();

      final S nextState = this.solutionProvider.generateExtendedSolution(this.encodingStrategy, currentState, action);

      response.setState(nextState);
      response.setAppliedActionId(action);

      function.evaluate((TransformationSolution) nextState);

      response.setReward(determineReward(nextState, true));
      response.setDone(determineIsEpisodeDone(nextState));

      currentState = nextState;

      return response;
   }

   @Override
   public SOEnvResponse<S> step(final LocalSearchStrategy strategy, final List<IApplicationState> action,
         final int explorationSteps) {
      final SOEnvResponse<S> response = new SOEnvResponse<>();

      int performedExplorationSteps = 0;
      S nextState = null;

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
               performedExplorationSteps = 1;
               evaluteSolution(nextState);
               break;
         }

      } else {
         nextState = this.solutionProvider.generateExtendedSolution(this.currentState, action);
         function.evaluate((TransformationSolution) nextState);
      }

      response.setNumExplorationSteps(performedExplorationSteps);
      response.setDone(determineIsEpisodeDone(nextState));

      if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
         response.setReward(determineReward(nextState, true));
         response.setState(nextState);
         currentState = nextState;

      }

      return response;
   }

   @Override
   public SOEnvResponse<S> step(final S solution, final INDArray actionProbs) {
      throw new RuntimeException("Not appropriate environment!");
   }

   @Override
   public SOEnvResponse<S> step(final S solution, final int action) {
      final SOEnvResponse<S> response = new SOEnvResponse<>();

      final S nextState = this.solutionProvider.generateExtendedSolution(this.encodingStrategy, solution, action);

      response.setState(nextState);
      response.setAppliedActionId(action);

      function.evaluate((TransformationSolution) nextState);

      response.setReward(determineReward(nextState, true));
      response.setDone(determineIsEpisodeDone(nextState));

      currentState = nextState;

      return response;
   }

   public SOEnvResponse<S> step(final String rule, final Map<String, Object> paramValues) {
      final SOEnvResponse<S> response = new SOEnvResponse<>();

      final S nextState = this.solutionProvider.generateExtendedSolution(currentState, rule, paramValues);

      if(nextState == null) {
         response.setState(null);
         return response;
      }
      response.setState(nextState);

      function.evaluate((TransformationSolution) nextState);

      final DoneStatus doneStatus = determineIsEpisodeDone(nextState);

      final double curFitness = determineReward(nextState, true);

      response.setReward(curFitness);
      response.setDone(doneStatus);

      currentState = nextState;

      return response;
   }
}
