package at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment;

import at.ac.tuwien.big.moea.search.algorithm.local.IFitnessComparator;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.LocalSearchStrategy;

import java.util.List;

import org.moeaframework.core.Solution;

public interface IMOEnvironment<S extends Solution> extends IEnvironment<S> {
   public List<IFitnessComparator<?, Solution>> getFitnessComparators();

   public List<String> getFunctionNames();

   double[] getRewards(final S state);

   public List<S> sample(int exploreSteps);

   public MOEnvResponse<S> step(LocalSearchStrategy localSearchStrategy, S currentSolution, int exploreSteps);

}
