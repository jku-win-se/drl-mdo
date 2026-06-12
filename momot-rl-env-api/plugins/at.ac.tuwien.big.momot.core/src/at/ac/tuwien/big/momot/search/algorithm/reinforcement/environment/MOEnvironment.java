package at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment;

import at.ac.tuwien.big.moea.problem.solution.variable.IPlaceholderVariable;
import at.ac.tuwien.big.moea.search.algorithm.local.IFitnessComparator;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.IMOEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.MOEnvResponse;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.LocalSearchStrategy;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.henshin.interpreter.UnitApplication;
import org.moeaframework.core.Solution;

public class MOEnvironment<S extends Solution> extends AbstractEnvironment<S> implements IMOEnvironment<S> {
   protected List<IFitnessComparator<?, Solution>> fitnessComparatorList;
   protected List<String> objectiveNames;
   protected IEGraphMultiDimensionalFitnessFunction f;

   public MOEnvironment(final IEGraphMultiDimensionalFitnessFunction f,
         final List<IFitnessComparator<?, Solution>> fitnessComparatorList, final List<String> objectiveNames,
         final IEncodingStrategy<S> encodingStrategy) {
      super(encodingStrategy);
      this.f = f;
      this.fitnessComparatorList = fitnessComparatorList;
      this.objectiveNames = objectiveNames;

   }

   private double[] determineRewards(final S state) {
      final int rewardDim = fitnessComparatorList.size();
      // final double[] prevObj = this.currentState.getObjectives();
      final double[] prevObjReward = new double[fitnessComparatorList.size()];

      // Arrays.parallelSetAll(prevObjReward, i -> prevObj[i] * -1);

      final double[] rewardVec = new double[rewardDim];
      for(int i = 0; i < fitnessComparatorList.size(); i++) {
         prevObjReward[i] = (Double) fitnessComparatorList.get(i).getValue(this.currentState) * -1;
         rewardVec[i] = (Double) fitnessComparatorList.get(i).getValue(state) * -1;
      }
      final double[] result = new double[rewardDim];
      Arrays.parallelSetAll(result, i -> rewardVec[i] - prevObjReward[i]);

      return result;
   }

   @Override
   public List<IFitnessComparator<?, Solution>> getFitnessComparators() {
      return this.fitnessComparatorList;
   }

   @Override
   public List<String> getFunctionNames() {
      return this.objectiveNames;
   }

   @Override
   public double[] getRewards(final S state) {
      return this.determineRewards(state);
   }

   private S localSearchMO(final List<S> solutions) {
      // double curMaxGain = Double.NEGATIVE_INFINITY;
      final Map<S, Double> scalRewards = new HashMap<>();
      for(final S s : solutions) {

         evaluteSolution(s);

         if(s.getNumberOfVariables() == this.currentState.getNumberOfVariables() || s.getNumberOfVariables() > 0
               && s.getVariable(s.getNumberOfVariables() - 1) instanceof IPlaceholderVariable) {
            continue;
         }
         final double[] solRewards = determineRewards(s);
         final double scalReward = Arrays.stream(solRewards).sum();
         scalRewards.put(s, scalReward);

      }
      return scalRewards.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
   }

   @Override
   public List<S> sample(final int exploreSteps) {
      final List<S> solutions = new ArrayList<>();
      for(final S solution : this.solutionProvider.generateNeighbors(currentState, exploreSteps, null)) {
         evaluteSolution(solution);
         solutions.add(solution);
      }
      return solutions;
   }

   @Override
   public MOEnvResponse<S> step(final LocalSearchStrategy strategy, final List<IApplicationState> action,
         final int explorationSteps) {
      final MOEnvResponse<S> response = new MOEnvResponse<>();

      int performedExplorationSteps = 0;
      S nextState = null;

      if(action == null) {

         switch(strategy) {
            case GREEDY:
               final List<S> solutions = new ArrayList<>();
               for(final S solution : this.solutionProvider.generateNeighbors(currentState, explorationSteps, null)) {
                  solutions.add(solution);
               }
               nextState = this.localSearchMO(solutions);
               performedExplorationSteps = solutions.size();
               break;
            case NONE:
               for(final S solution : this.solutionProvider.generateNeighbors(currentState, 1, null)) {
                  nextState = solution;
               }

               evaluteSolution(nextState);
               final UnitApplication ua = (UnitApplication) nextState.getVariable(nextState.getNumberOfVariables() - 1);

               // System.out.println(ua.getUnit().getName());
               // for(final Parameter p : ua.getUnit().getParameters()) {
               // System.out.println(p.getName() + " " + ua.getAssignment().getParameterValue(p));
               //
               // }

               performedExplorationSteps = 1;
               break;
         }

      } else {
         nextState = this.solutionProvider.generateExtendedSolution(this.currentState, action);
         f.evaluate((TransformationSolution) nextState);
      }

      response.setNumExplorationSteps(performedExplorationSteps);
      response.setDone(determineIsEpisodeDone(nextState));

      if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
         response.setRewards(determineRewards(nextState));
         response.setState(nextState);
         currentState = nextState;
      }

      return response;
   }

   @Override
   public MOEnvResponse<S> step(final LocalSearchStrategy strategy, final S currentSolution,
         final int explorationSteps) {
      final MOEnvResponse<S> response = new MOEnvResponse<>();

      S nextState = null;

      switch(strategy) {
         case GREEDY:
            final List<S> solutions = new ArrayList<>();
            for(final S solution : this.solutionProvider.generateNeighbors(currentSolution, explorationSteps, null)) {
               solutions.add(solution);
            }
            nextState = this.localSearchMO(solutions);
            break;
         case NONE:
            for(final S solution : this.solutionProvider.generateNeighbors(currentSolution, 1, null)) {
               nextState = solution;
            }
            evaluteSolution(nextState);
            break;
      }

      response.setDone(determineIsEpisodeDone(nextState));

      if(nextState != null && nextState.getNumberOfVariables() > currentState.getNumberOfVariables()) {
         response.setRewards(determineRewards(nextState));
         response.setState(nextState);
         currentState = nextState;
      }

      return response;
   }

}
