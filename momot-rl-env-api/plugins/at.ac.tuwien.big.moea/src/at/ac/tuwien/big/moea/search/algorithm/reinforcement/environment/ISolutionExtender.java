package at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment;

import at.ac.tuwien.big.moea.search.algorithm.local.INeighborhood;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;

import java.util.List;
import java.util.Map;

import org.eclipse.emf.henshin.model.Unit;
import org.moeaframework.core.Solution;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface ISolutionExtender<S extends Solution> {

   Object[] generateExtendedSolution(final IEncodingStrategy<S> encoder, final S solution, final INDArray distribution);

   S generateExtendedSolution(final IEncodingStrategy<S> encoder, S solution, int action);

   S generateExtendedSolution(S currentState, List<IApplicationState> action);

   S generateExtendedSolution(S solution, String rule, Map<String, Object> paramValues);

   default S generateExtendedSolutionInPlace(final S solution, final String rule,
         final Map<String, Object> paramValues) {
      return generateExtendedSolution(solution, rule, paramValues);
   }

   INeighborhood<S> generateNeighbors(S solution, int maxNeighbors, IEncodingStrategy<S> encodingStrategy);

   Map<String, Unit> getUnitMapping();

}
