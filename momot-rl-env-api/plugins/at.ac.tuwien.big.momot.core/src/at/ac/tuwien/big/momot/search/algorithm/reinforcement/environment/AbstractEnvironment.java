package at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.IEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.ISolutionExtender;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.IRLUtils;
import at.ac.tuwien.big.momot.problem.solution.variable.ITransformationVariable;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.algorithm.RLUtils;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.henshin.model.Unit;
import org.moeaframework.core.Solution;
import org.moeaframework.core.Variable;

public abstract class AbstractEnvironment<S extends Solution> implements IEnvironment<S> {

   protected static final boolean USE_ABSOLUTE_COST = false;
   protected S initialState = null;
   protected S currentState = null;
   protected ISolutionExtender<S> solutionProvider;
   protected int maxSolutionLength;

   protected final IEncodingStrategy<S> encodingStrategy;
   protected double mu; // penalty multiplier
   private final double muMin; // lower bound
   private final double muMax; // upper bound
   private final double etaMu; // adaptation rate
   private final double decayFactor; // how much to shrink mu when feasible
   private final double fTarget; // target feasibility rate
   protected final int windowSize; // K episodes for moving average

   protected final List<Boolean> feasibilityWindow;
   protected double currentEpisodeViolation = 0.0;
   protected final boolean useLagrangian = false;
   protected final IRLUtils<S> utils;
   protected boolean foundFeasibleState = false;
   private Object agentClass = null;
   private Method evaluateFunction = null;
   protected double initialCost;

   public AbstractEnvironment(final IEncodingStrategy<S> encodingStrategy) {
      this.utils = new RLUtils<>();
      this.encodingStrategy = encodingStrategy;
      this.mu = 1;
      this.muMin = 0.01;
      this.muMax = 10;
      this.etaMu = .01;
      this.fTarget = 1.0;
      this.windowSize = 20;
      this.decayFactor = .9;
      this.feasibilityWindow = new ArrayList<>();
   }

   public void adaptLagragian() {
      if(useLagrangian) {
         if(feasibilityWindow.size() == 0) {
            throw new RuntimeException("No entries in feasibility window!");
         }
         // if(feasibilityWindow.size() > windowSize) {
         // feasibilityWindow.removeFirst();
         // }

         final long feasibleCount = feasibilityWindow.stream().filter(f -> f).count();
         final double fRate = (double) feasibleCount / feasibilityWindow.size();

         mu = Math.max(muMin, 1 - fRate);

         System.out.println("New mu: " + mu + " (from fRate = " + fRate + ")");

         feasibilityWindow.clear();
         currentEpisodeViolation = 0.0;

         // Lagrangian update
         // if(fRate >= fTarget) {
         // // Agent consistently finds feasible solutions → scale down μ
         // mu = Math.max(muMin, mu * decayFactor);
         // } else {
         // // Not yet reliably feasible → increase μ slightly
         // mu = Math.min(muMax, mu + etaMu * (fTarget - fRate));
         // }

         // // Optional: safety rollback if violation spikes
         // // (simple heuristic)
         // if(meanViolation > 2 * violationTarget) {
         // mu *= 1.5;
         // }

         // Reset episode violation

      }
   }

   public DoneStatus determineIsEpisodeDone(final S nextState) {

      if(nextState == null || determineIsFinalState(nextState)) {
         return DoneStatus.FINAL_STATE_REACHED;
      }

      if(encodingStrategy != null && (determineIsEpisodeDone(nextState, encodingStrategy)
            || encodingStrategy.isEpisodeEndingState(nextState))) {
         return DoneStatus.FINAL_STATE_REACHED;
      }

      if(nextState.getNumberOfVariables() >= this.maxSolutionLength) {
         return DoneStatus.MAX_LENGTH_REACHED;
      }

      return null;
   }

   private boolean determineIsEpisodeDone(final S s, final IEncodingStrategy<S> encoder) {
      final List<String> endingRules = encoder.getEpisodeEndingRules();
      if(endingRules != null) {
         for(int i = 0; i < s.getNumberOfVariables(); i++) {
            final Variable var = s.getVariable(i);
            if(endingRules.contains(((ITransformationVariable) var).getUnit().getName())) {
               return true;
            }
         }
      }
      return false;
   }

   protected boolean determineIsFinalState(final S nextState) {
      return this.currentState.getNumberOfVariables() == nextState.getNumberOfVariables();
   }

   public void endEpisode() {

      this.feasibilityWindow.add(foundFeasibleState);

   }

   public void evaluteSolution(final S state) {
      try {
         evaluateFunction.invoke(agentClass, state);
      } catch(IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
         e.printStackTrace();
      }
   }

   public double getFeasibilityRate() {
      final long feasibleCount = feasibilityWindow.stream().filter(f -> f).count();
      return (double) feasibleCount / Math.max(1, feasibilityWindow.size());
   }

   @Override
   public S getInitialSolution() {
      return initialState;
   }

   @Override
   public int getMaxSolutionLength() {
      return this.maxSolutionLength;
   }

   @Override
   public IEncodingStrategy<S> getProblemEncoder() {
      return this.encodingStrategy;
   }

   @Override
   public IRLUtils<S> getRLUtils() {
      return this.utils;
   }

   @Override
   public Map<String, Unit> getUnitMapping() {
      return this.solutionProvider.getUnitMapping();
   }

   @Override
   public S reset() {
      if(!utils.isEvaluated(initialState)) {
         if(this.evaluateFunction != null) {
            this.evaluteSolution(initialState);
         } else {
            throw new RuntimeException("Initial state not evaluated but evaluation function is null!");
         }
      }

      currentState = utils.newTransformationSolution(initialState);

      if(useLagrangian) {
         currentEpisodeViolation = .0;
      }

      return currentState;
   }

   @Override
   public void setEvaluationMethod(final Object agentInstance, final Method evaluateFunction) {
      this.agentClass = agentInstance;
      this.evaluateFunction = evaluateFunction;
   }

   // ONLY USED FROM BaseEnvironment
   @Override
   public void setInitialSolution(final S solution) {
      // this.initialState = utils.newTransformationSolution(solution);
      // this.initialState.setObjectives(solution.getObjectives());
      // this.initialState.setConstraints(solution.getConstraints());
      this.initialState = utils.newTransformationSolution(solution);

   }

   @Override
   public void setSolutionLength(final int length) {
      this.maxSolutionLength = length;
   }

   @Override
   public void setSolutionProvider(final ISolutionExtender<S> provider) {
      this.solutionProvider = provider;
   }

}
