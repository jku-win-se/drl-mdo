package at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment;

import at.ac.tuwien.big.moea.problem.solution.variable.IPlaceholderVariable;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IRewardStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.ISOEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.LocalSearchStrategy;
import at.ac.tuwien.big.moea.search.fitness.MultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.ToDoubleFunction;

import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.henshin.interpreter.Engine;
import org.moeaframework.core.Solution;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SOEnvironment<S extends Solution> extends AbstractEnvironment<S> implements ISOEnvironment<S> {

   @SuppressWarnings("unused")
   private final IRewardStrategy<S> rewardStrategy;
   private double prevFitness = .0;
   private double prevConstraintViolation = .0;
   private boolean applyRulesInPlace = false;

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
      return this.determineRewardByFitnessFunction(nextState, updatePrevs);
   }

   private double calculateConstraintReward(final double constraintVal) {
      return -(constraintVal - prevConstraintViolation);
   }

   private double calculateFitnessReward(final double fitnessVal) {
      return -(fitnessVal - prevFitness);
   }

   private double determineRewardByFitnessFunction(final S state, final boolean updatePrevs) {
      double totalReward = 0.0;
      final boolean wasFeasible = this.constraintAggregator == null || prevConstraintViolation == 0.0;
      boolean isFeasible = wasFeasible;

      if(this.constraintAggregator != null) {
         final double constraintVal = this.constraintAggregator.applyAsDouble(this.retrieveConstraints(state));
         isFeasible = constraintVal == 0.0;

         if(!wasFeasible || !isFeasible) {
            totalReward += calculateConstraintReward(constraintVal)
                  * (this.constraintScaler != null ? this.getConstraintScaler(state) : 1.);
         }

         if(updatePrevs) {
            prevConstraintViolation = constraintVal;
         }
      }

      final double[] objectives = this.retrieveObjectives(state);
      final double fitnessVal = this.fitnessAggregator.applyAsDouble(objectives);

      if(wasFeasible && isFeasible && Double.isFinite(fitnessVal)) {
         totalReward += calculateFitnessReward(fitnessVal)
               * (this.fitnessScaler != null ? this.getFitnessScaler(state) : 1.);
      }

      if(updatePrevs && isFeasible && Double.isFinite(fitnessVal)) {
         prevFitness = fitnessVal;
      }

      return totalReward;
   }

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

         final double nextStateReward = calcReward(solution, false);

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

      if(this.constraintAggregator != null) {
         prevConstraintViolation = this.constraintAggregator
               .applyAsDouble(this.adaptInitialConstraints(currentState.getConstraints()));

         if(Double.isNaN(prevConstraintViolation) || Math.abs(prevConstraintViolation) == Double.POSITIVE_INFINITY) {
            throw new RuntimeException("Cannot evalute constraint violations: Turns out as NaN/Infinity value");
         }
      } else {
         prevConstraintViolation = 0.0;
      }

      final double initialFitness = this.fitnessAggregator
            .applyAsDouble(this.adaptInitialObjectives(currentState.getObjectives()));

      if(Double.isFinite(initialFitness)) {
         prevFitness = initialFitness;
      } else {
         prevFitness = 0;
      }

      return currentState;
   }

   private double[] retrieveConstraints(final S state) {
      final double[] constraints = new double[state.getNumberOfConstraints()];
      for(int i = 0; i < constraints.length; i++) {
         constraints[i] = (double) state.getAttribute(MultiDimensionalFitnessFunction.constraintKey(i));
         if(this.constraintNormalizers != null && i < this.constraintNormalizers.length) {
            constraints[i] *= this.constraintNormalizers[i];
         }
      }

      return constraints;
   }

   public double[] retrieveObjectives(final S state) {
      final double[] objectives = new double[state.getNumberOfObjectives()];
      for(int i = 0; i < objectives.length; i++) {
         objectives[i] = (double) state.getAttribute(MultiDimensionalFitnessFunction.objectiveKey(i));
         if(this.fitnessNormalizers != null && i < this.fitnessNormalizers.length) {
            objectives[i] *= this.fitnessNormalizers[i];
         }
      }

      return objectives;
   }

   public void setApplyRulesInPlace(final boolean applyRulesInPlace) {
      this.applyRulesInPlace = applyRulesInPlace;
   }

   public void setConstraintNormalizers(final double[] constraintNormalizers) {
      this.constraintNormalizers = constraintNormalizers;
   }

   public void setFitnessNormalizers(final double[] fitnessNormalizers) {
      this.fitnessNormalizers = fitnessNormalizers;
   }

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
   }

   @Override
   public SOEnvResponse<S> step(final S solution, final int action) {
      throw new RuntimeException("Not implemented");
   }

   public SOEnvResponse<S> step(final String rule, final Map<String, Object> paramValues) {
      final SOEnvResponse<S> response = new SOEnvResponse<>();

      final S nextState = applyRulesInPlace
            ? this.solutionProvider.generateExtendedSolutionInPlace(currentState, rule, paramValues)
            : this.solutionProvider.generateExtendedSolution(currentState, rule, paramValues);

      if(nextState == null) {
         response.setState(null);
         response.setReward(0.0);
         response.setDone(determineIsEpisodeDone(null));
         return response;
      }

      evaluteSolution(nextState);

      final DoneStatus doneStatus = determineIsEpisodeDone(nextState);

      response.setDone(doneStatus);

      final double reward = calcReward(nextState, true);

      response.setReward(reward);
      if(nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
         response.setState(nextState);
         currentState = nextState;
      }

      return response;
   }

}
