//package at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment;
//
//import at.ac.tuwien.big.moea.problem.solution.variable.IPlaceholderVariable;
//import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
//import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;
//import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IRewardStrategy;
//import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
//import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.ISOEnvironment;
//import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
//import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.LocalSearchStrategy;
//
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Map;
//import java.util.function.ToDoubleFunction;
//
//import org.eclipse.emf.henshin.interpreter.RuleApplication;
//import org.moeaframework.core.Solution;
//import org.nd4j.linalg.api.ndarray.INDArray;
//
//public class SOEnvironment<S extends Solution> extends AbstractEnvironment<S> implements ISOEnvironment<S> {
//
//   private final IRewardStrategy<S> rewardStrategy;
//   private double prevFitness = .0;
//   private double prevConstraintViolation = .0;
//   private boolean fitnessEverEvaluated = false; // Add this
//   private final boolean debug = false;
//
//   ToDoubleFunction<double[]> fitnessAggregator;
//   ToDoubleFunction<double[]> constraintAggregator;
//
//   public SOEnvironment(final ToDoubleFunction<double[]> fitnessAggregator,
//         final ToDoubleFunction<double[]> constraintAggregator, final IRewardStrategy<S> rewardStrategy,
//         final IEncodingStrategy<S> encodingStrategy) {
//      super(encodingStrategy);
//      this.rewardStrategy = rewardStrategy;
//      this.fitnessAggregator = fitnessAggregator;
//      this.constraintAggregator = constraintAggregator;
//
//   }
//
//   private double determineReward(final S state, final boolean updatePrevs) {
//      if(this.rewardStrategy != null && this.rewardStrategy.getRewardMap() != null) {
//
//         return this.determineRewardByTransformations(this.utils.getApplicationStatesDiff(currentState, state))
//               + this.rewardStrategy.determineAdditionalReward(currentState, state);
//      }
//      return this.determineRewardByFitnessFunction(state, updatePrevs);
//   }
//
//   private double determineRewardByFitnessFunction(final S state, final boolean updatePrevs) {
//      double totalReward = 0.0;
//
//      // === Constraint Evaluation ===
//      if(this.constraintAggregator != null) {
//         final double constraintVal = this.constraintAggregator.applyAsDouble(this.retrieveConstraints(state));
//
//         // Absolute reward regardless of previous feasibility
//         totalReward += -constraintVal;
//
//         if(debug) {
//            System.out.println("constraintReward: " + totalReward);
//         }
//
//      }
//
//      // === Fitness Evaluation ===
//      final double[] objectives = this.retrieveObjectives(state);
//      final Double fitnessValObj = this.fitnessAggregator.applyAsDouble(objectives);
//
//      if(Math.abs(fitnessValObj) != Double.POSITIVE_INFINITY && !Double.isNaN(fitnessValObj)) {
//         final double fitnessVal = fitnessValObj;
//
//         // if(fitnessEverEvaluated) {
//         totalReward += -fitnessVal;
//         // } else if(updatePrevs) {
//         // if(debug) {
//         // System.out.println("Fitness eval true");
//         // }
//         // fitnessEverEvaluated = true;
//         // }
//
//         if(updatePrevs) {
//            if(debug) {
//               System.out.println("fitnessReward: " + -fitnessVal);
//            }
//         }
//      }
//
//      return totalReward;
//   }
//
//   private double determineRewardByTransformations(final List<IApplicationState> ruleAssignments) {
//      double reward = 0;
//      for(final IApplicationState actionRule : ruleAssignments) {
//         final String ruleName = actionRule.getUnit().getName();
//         for(final RuleApplication rule : actionRule.getAppliedRules()) {
//            if(rewardStrategy.getRewardMap().containsKey(rule.getRule().getName())) {
//               reward += rewardStrategy.getRewardMap().get(rule.getRule().getName());
//            }
//         }
//      }
//
//      return reward;
//   }
//
//   @Override
//   public String getFunctionName() {
//      throw new RuntimeException("Not implemented");
//   }
//
//   @Override
//   public IEncodingStrategy<S> getProblemEncoder() {
//      return this.encodingStrategy;
//   }
//
//   @Override
//   public double getReward(final S state) {
//      return this.determineReward(state, true);
//   }
//
//   @Override
//   public SOEnvResponse<S> inferStep(final List<IApplicationState> action) {
//      final SOEnvResponse<S> response = new SOEnvResponse<>();
//
//      S nextState = null;
//
//      nextState = this.solutionProvider.generateExtendedSolution(this.currentState, action);
//
//      evaluteSolution(nextState);
//
//      response.setDone(determineIsEpisodeDone(nextState));
//
//      if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
//         response.setState(nextState);
//         currentState = nextState;
//      }
//
//      return response;
//   }
//
//   private S localSearchSO(final List<S> solutions) {
//      double maxReward = Double.NEGATIVE_INFINITY;
//      S nextState = null;
//
//      for(final S solution : solutions) {
//
//         evaluteSolution(solution);
//
//         if(solution.getNumberOfVariables() == this.currentState.getNumberOfVariables()
//               || solution.getNumberOfVariables() > 0
//                     && solution.getVariable(solution.getNumberOfVariables() - 1) instanceof IPlaceholderVariable) {
//            continue;
//         }
//
//         final double nextStateReward = determineReward(solution, false);
//
//         if(nextStateReward > maxReward) {
//            maxReward = nextStateReward;
//            nextState = solution;
//         }
//      }
//
//      return nextState;
//   }
//
//   @Override
//   public S reset() {
//      currentState = super.reset();
//
//      final double initialFitness = this.fitnessAggregator.applyAsDouble(currentState.getObjectives());
//
//      // final double initialFitness = (Double) fitnessComparator.getValue(currentState);
//
//      if(!Double.isNaN(initialFitness) && Math.abs(initialFitness) != Double.POSITIVE_INFINITY) {
//         fitnessEverEvaluated = true;
//         prevFitness = initialFitness;
//      } else {
//         fitnessEverEvaluated = false;
//         prevFitness = 0;
//      }
//
//      if(this.constraintAggregator != null) {
//         prevConstraintViolation = this.constraintAggregator.applyAsDouble(currentState.getConstraints());
//
//         if(Double.isNaN(prevConstraintViolation) || Math.abs(prevConstraintViolation) == Double.POSITIVE_INFINITY) {
//            throw new RuntimeException("Cannot evalute constraint violations: Turns out as NaN/Infinity value");
//         }
//      }
//
//      return currentState;
//   }
//
//   private double[] retrieveConstraints(final S state) {
//      final double[] constraints = new double[state.getNumberOfConstraints()];
//      for(int i = 0; i < constraints.length; i++) {
//         constraints[i] = (double) state.getAttribute("c_dim_" + String.valueOf(i));
//      }
//
//      return constraints;
//   }
//
//   private double[] retrieveObjectives(final S state) {
//      final double[] objectives = new double[state.getNumberOfObjectives()];
//      for(int i = 0; i < objectives.length; i++) {
//         objectives[i] = (double) state.getAttribute("f_dim_" + String.valueOf(i));
//      }
//
//      return objectives;
//   }
//
//   @Override
//   public SOEnvResponse<S> step(final LocalSearchStrategy strategy, final List<IApplicationState> action,
//         final int explorationSteps) {
//      final SOEnvResponse<S> response = new SOEnvResponse<>();
//
//      S nextState = null;
//      int performedExplorationSteps = 0;
//
//      if(action == null) {
//
//         switch(strategy) {
//            case GREEDY:
//               final List<S> solutions = new ArrayList<>();
//
//               for(final S solution : this.solutionProvider.generateNeighbors(currentState, explorationSteps,
//                     this.encodingStrategy)) {
//                  solutions.add(solution);
//               }
//               performedExplorationSteps = solutions.size();
//               nextState = this.localSearchSO(solutions);
//               break;
//            case NONE:
//               for(final S solution : this.solutionProvider.generateNeighbors(currentState, 1, this.encodingStrategy)) {
//                  nextState = solution;
//               }
//               evaluteSolution(nextState);
//               performedExplorationSteps = 1;
//               break;
//         }
//
//      } else {
//         nextState = this.solutionProvider.generateExtendedSolution(this.currentState, action);
//         evaluteSolution(nextState);
//      }
//
//      response.setDone(determineIsEpisodeDone(nextState));
//      response.setNumExplorationSteps(performedExplorationSteps);
//
//      if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
//         response.setReward(determineReward(nextState, true));
//         response.setState(nextState);
//         currentState = nextState;
//
//      }
//      return response;
//   }
//
//   @Override
//   public SOEnvResponse<S> step(final S solution, final INDArray actionProbs) {
//      final SOEnvResponse<S> response = new SOEnvResponse<>();
//
//      final Object[] actionSolution = this.solutionProvider.generateExtendedSolution(this.encodingStrategy, solution,
//            actionProbs);
//      final int action = (int) actionSolution[0];
//      final S nextState = (S) actionSolution[1];
//
//      response.setState(nextState);
//      response.setAppliedActionId(action);
//
//      evaluteSolution(nextState);
//
//      response.setReward(determineReward(nextState, true));
//      response.setDone(determineIsEpisodeDone(nextState));
//
//      currentState = nextState;
//
//      return response;
//   }
//
//   // public SOEnvResponse<S> step(final String rule, final Map<String, Object> paramValues) {
//   // final SOEnvResponse<S> response = new SOEnvResponse<>();
//   //
//   // final S nextState = this.solutionProvider.generateExtendedSolution(currentState, rule, paramValues);
//   //
//   // if(nextState == null) {
//   // response.setState(null);
//   // return response;
//   // }
//   // response.setState(nextState);
//   //
//   // function.evaluate((TransformationSolution) nextState);
//   //
//   //
//   //
//   // final DoneStatus doneStatus = determineIsEpisodeDone(nextState);
//   //
//   // final double curFitness = determineReward(nextState, true);
//   //
//   // response.setReward(curFitness);
//   // response.setDone(doneStatus);
//   //
//   // currentState = nextState;
//   //
//   // return response;
//   // }
//
//   @Override
//   public SOEnvResponse<S> step(final S solution, final int action) {
//      final SOEnvResponse<S> response = new SOEnvResponse<>();
//
//      final S nextState = this.solutionProvider.generateExtendedSolution(this.encodingStrategy, solution, action);
//
//      response.setState(nextState);
//      response.setAppliedActionId(action);
//
//      evaluteSolution(nextState);
//
//      response.setReward(determineReward(nextState, true));
//      response.setDone(determineIsEpisodeDone(nextState));
//
//      currentState = nextState;
//
//      return response;
//   }
//
//   public SOEnvResponse<S> step(final String rule, final Map<String, Object> paramValues) {
//      final SOEnvResponse<S> response = new SOEnvResponse<>();
//
//      final S nextState = this.solutionProvider.generateExtendedSolution(currentState, rule, paramValues);
//
//      evaluteSolution(nextState);
//
//      final DoneStatus doneStatus = determineIsEpisodeDone(nextState);
//
//      response.setDone(doneStatus);
//
//      if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
//         response.setReward(determineReward(nextState, true));
//         response.setState(nextState);
//         currentState = nextState;
//
//      }
//
//      return response;
//   }
//
//}


