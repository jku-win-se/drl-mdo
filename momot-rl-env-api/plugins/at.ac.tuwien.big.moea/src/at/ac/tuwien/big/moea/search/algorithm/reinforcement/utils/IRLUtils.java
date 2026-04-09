package at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IMOQTableAccessor;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IParetoQTableAccessor;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IQTableAccessor;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.ISOQTableAccessor;

import java.util.List;
import java.util.Map;

import org.eclipse.emf.henshin.model.Unit;
import org.moeaframework.core.Solution;

public interface IRLUtils<S extends Solution> {

   public S copySolution(S solution, boolean execute);

   public List<IApplicationState> getApplicationStates(final S solution);

   public List<IApplicationState> getApplicationStatesDiff(final S cur, final S next);

   IMOQTableAccessor<List<IApplicationState>, List<IApplicationState>> initMOQTable(Map<String, Unit> unitMapping);

   IParetoQTableAccessor<List<IApplicationState>, List<IApplicationState>> initParetoQTable(
         Map<String, Unit> unitMapping);

   ISOQTableAccessor<List<IApplicationState>, List<IApplicationState>> initSOQTable(Map<String, Unit> unitMapping);

   boolean isEvaluated(final S s);

   IMOQTableAccessor<List<IApplicationState>, List<IApplicationState>> loadMOQTable(final String inputSrc,
         Map<String, Unit> unitMapping);

   IParetoQTableAccessor<List<IApplicationState>, List<IApplicationState>> loadParetoQTable(String qTableIn,
         Map<String, Unit> unitMapping);

   ISOQTableAccessor<List<IApplicationState>, List<IApplicationState>> loadSOQTable(final String inputSrc,
         Map<String, Unit> unitMapping);

   S newTransformationSolution(final S s);

   void writeQTableToDisk(final IQTableAccessor<List<IApplicationState>, List<IApplicationState>> qTable,
         final String outputSrc);

}
