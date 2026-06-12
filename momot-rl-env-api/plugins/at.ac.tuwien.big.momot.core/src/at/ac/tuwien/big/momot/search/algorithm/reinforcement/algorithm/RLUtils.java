package at.ac.tuwien.big.momot.search.algorithm.reinforcement.algorithm;

import at.ac.tuwien.big.moea.problem.solution.variable.IPlaceholderVariable;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IMOQTableAccessor;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IParetoQTableAccessor;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IQTableAccessor;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.ISOQTableAccessor;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.ParetoQState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.IRLUtils;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.problem.solution.variable.RuleApplicationVariable;
import at.ac.tuwien.big.momot.problem.solution.variable.UnitApplicationVariable;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.ApplicationState;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.MOQTable;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.ParetoQTable;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.SOQTable;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.eclipse.emf.henshin.interpreter.RuleApplication;
import org.eclipse.emf.henshin.interpreter.UnitApplication;
import org.eclipse.emf.henshin.model.Unit;
import org.moeaframework.core.Solution;
import org.moeaframework.core.Variable;

public class RLUtils<S extends Solution> implements IRLUtils<S> {

   // private Map<String, Object> extractParameters(final Assignment assignment) {
   // final Map<String, Object> paramValues = new HashMap<>();
   //
   // for(final Parameter p : assignment.getUnit().getParameters()) {
   // if(assignment.getParameterValue(p) != null) {
   // paramValues.put(p.getName(), assignment.getParameterValue(p));
   // }
   // }
   //
   // return paramValues;
   // }
   //
   // private Map<String, Object> extractParameters(final RuleApplication... raps) {
   // final Map<String, Object> paramValues = new HashMap<>();
   //
   // for(final RuleApplication ra : raps) {
   // for(final Parameter p : ra.getUnit().getParameters()) {
   // if(ra.getPartialMatch().getParameterValue(p) != null) {
   // paramValues.put(p.getName(), ra.getPartialMatch().getParameterValue(p));
   // }
   // }
   //
   // }
   // return paramValues;
   // }

   // @Override
   // public ISolutionExtender<TransformationSolution> createSolutionProvider() {
   // final Engine engine = new MomotEngine(true, new String());
   //
   // return new SolutionProvider<S>(new SearchHelper(engine));
   // }

   @Override
   public S copySolution(final S s, final boolean execute) {
      final TransformationSolution ts = (TransformationSolution) s;
      final TransformationSolution newTs = new TransformationSolution(MomotUtil.copy(ts.getSourceGraph()),
            ts.getNumberOfVariables(), ts.getNumberOfObjectives(), ts.getNumberOfConstraints());
      newTs.setTransformation(new ArrayList<>(Arrays.asList(ts.getVariables())), null);
      if(execute) {
         newTs.execute();
      }
      newTs.setObjectives(s.getObjectives());
      newTs.setConstraints(s.getConstraints());

      return (S) newTs;
   }

   @Override
   public List<IApplicationState> getApplicationStates(final S solution) {

      final List<IApplicationState> stateList = new ArrayList<>();

      for(int i = 0; i < solution.getNumberOfVariables(); i++) {
         final UnitApplication ua = (UnitApplication) solution.getVariable(i);
         stateList.add(unitApplicationToApplicationState(ua));
      }

      return stateList;
   }

   @Override
   public List<IApplicationState> getApplicationStatesDiff(final S cur, final S next) {
      final List<UnitApplication> trafoVars1 = new ArrayList<>();
      final List<UnitApplication> trafoVars2 = new ArrayList<>();
      for(int i = 0; i < cur.getNumberOfVariables(); i++) {
         final Variable v = cur.getVariable(i);
         if(!(v instanceof IPlaceholderVariable)) {
            trafoVars1.add((UnitApplication) v);
         }

      }

      for(int i = 0; i < next.getNumberOfVariables(); i++) {
         final Variable v = next.getVariable(i);

         if(!(v instanceof IPlaceholderVariable)) {
            trafoVars2.add((UnitApplication) v);
         }
      }

      final List<IApplicationState> stateList = new ArrayList<>();

      for(final UnitApplication ua : trafoVars2.subList(trafoVars1.size(), trafoVars2.size())) {
         stateList.add(unitApplicationToApplicationState(ua));
      }
      return stateList;
   }

   @Override
   public IMOQTableAccessor<List<IApplicationState>, List<IApplicationState>> initMOQTable(
         final Map<String, Unit> unitMapping) {
      return new MOQTable<>();
   }

   @Override
   public IParetoQTableAccessor<List<IApplicationState>, List<IApplicationState>> initParetoQTable(
         final Map<String, Unit> unitMapping) {
      return new ParetoQTable<>();
   }

   @Override
   public ISOQTableAccessor<List<IApplicationState>, List<IApplicationState>> initSOQTable(
         final Map<String, Unit> unitMapping) {
      return new SOQTable<>();
   }

   @Override
   public boolean isEvaluated(final S s) {
      final TransformationSolution ts = (TransformationSolution) s;
      return ts.getResultGraph() != null;
   }

   @Override
   public IMOQTableAccessor<List<IApplicationState>, List<IApplicationState>> loadMOQTable(final String inputSrc,
         final Map<String, Unit> unitMapping) {
      MOQTable<List<IApplicationState>, List<IApplicationState>> o = null;
      FileInputStream fis = null;
      ObjectInputStream in = null;
      try {
         fis = new FileInputStream(inputSrc);
         in = new ObjectInputStream(fis);
         o = (MOQTable<List<IApplicationState>, List<IApplicationState>>) in.readObject();
         in.close();
      } catch(final IOException ex) {
         ex.printStackTrace();
      } catch(final ClassNotFoundException ex) {
         ex.printStackTrace();
      }

      final Map<String, Unit> unitPlainNameMapping = new HashMap<>();
      for(final Entry<String, Unit> e : unitMapping.entrySet()) {
         final String[] strings = e.getKey().split("::");
         unitPlainNameMapping.put(strings[strings.length - 1], e.getValue());
      }

      for(final Entry<List<IApplicationState>, Map<List<IApplicationState>, double[]>> e : o.getTable().entrySet()) {
         final List<IApplicationState> aList = e.getKey();
         for(final IApplicationState as : aList) {
            as.setUnit(unitPlainNameMapping.get(as.getUnitName()));
         }
         for(final Entry<List<IApplicationState>, double[]> e2 : e.getValue().entrySet()) {
            for(final IApplicationState as2 : e2.getKey()) {
               as2.setUnit(unitPlainNameMapping.get(as2.getUnitName()));
            }
         }

      }

      return o;
   }

   @Override
   public IParetoQTableAccessor<List<IApplicationState>, List<IApplicationState>> loadParetoQTable(
         final String inputSrc, final Map<String, Unit> unitMapping) {

      ParetoQTable<List<IApplicationState>, List<IApplicationState>> o = null;
      FileInputStream fis = null;
      ObjectInputStream in = null;
      try {
         fis = new FileInputStream(inputSrc);
         in = new ObjectInputStream(fis);
         o = (ParetoQTable<List<IApplicationState>, List<IApplicationState>>) in.readObject();
         in.close();
      } catch(final IOException ex) {
         ex.printStackTrace();
      } catch(final ClassNotFoundException ex) {
         ex.printStackTrace();
      }

      final Map<String, Unit> unitPlainNameMapping = new HashMap<>();
      for(final Entry<String, Unit> e : unitMapping.entrySet()) {
         final String[] strings = e.getKey().split("::");
         unitPlainNameMapping.put(strings[strings.length - 1], e.getValue());
      }

      for(final Entry<List<IApplicationState>, Map<List<IApplicationState>, ParetoQState>> e : o.getTable()
            .entrySet()) {
         final List<IApplicationState> aList = e.getKey();
         for(final IApplicationState as : aList) {
            as.setUnit(unitPlainNameMapping.get(as.getUnitName()));
         }
         for(final Entry<List<IApplicationState>, ParetoQState> e2 : e.getValue().entrySet()) {
            for(final IApplicationState as2 : e2.getKey()) {
               as2.setUnit(unitPlainNameMapping.get(as2.getUnitName()));
            }
         }

      }

      return o;
   }

   @Override
   public ISOQTableAccessor<List<IApplicationState>, List<IApplicationState>> loadSOQTable(final String inputSrc,
         final Map<String, Unit> unitMapping) {
      SOQTable<List<IApplicationState>, List<IApplicationState>> o = null;
      FileInputStream fis = null;
      ObjectInputStream in = null;
      try {
         fis = new FileInputStream(inputSrc);
         in = new ObjectInputStream(fis);
         o = (SOQTable<List<IApplicationState>, List<IApplicationState>>) in.readObject();
         in.close();
      } catch(final IOException ex) {
         ex.printStackTrace();
      } catch(final ClassNotFoundException ex) {
         ex.printStackTrace();
      }

      final Map<String, Unit> unitPlainNameMapping = new HashMap<>();
      for(final Entry<String, Unit> e : unitMapping.entrySet()) {
         final String[] strings = e.getKey().split("::");
         unitPlainNameMapping.put(strings[strings.length - 1], e.getValue());
      }

      for(final Entry<List<IApplicationState>, Map<List<IApplicationState>, Double>> e : o.getTable().entrySet()) {
         final List<IApplicationState> aList = e.getKey();
         for(final IApplicationState as : aList) {
            as.setUnit(unitPlainNameMapping.get(as.getUnitName()));
         }
         for(final Entry<List<IApplicationState>, Double> e2 : e.getValue().entrySet()) {
            for(final IApplicationState as2 : e2.getKey()) {
               as2.setUnit(unitPlainNameMapping.get(as2.getUnitName()));
            }
         }

      }

      return o;
   }

   @Override
   public S newTransformationSolution(final S s) {
      final TransformationSolution ts = (TransformationSolution) s;
      // final TransformationSolution newTs = new TransformationSolution(MomotUtil.copy(ts.getSourceGraph()),
      // ts.getNumberOfVariables(), ts.getNumberOfObjectives(), ts.getNumberOfConstraints());
      // newTs.setTransformation(new ArrayList<>(Arrays.asList(ts.getVariables())), null);
      // newTs.execute();
      // newTs.setObjectives(s.getObjectives());
      // newTs.setConstraints(s.getConstraints());

      final TransformationSolution newTs = ts.copy();
      newTs.execute();

      return (S) newTs;
   }

   private IApplicationState unitApplicationToApplicationState(final UnitApplication ua) {
      Map<String, Object> params = new HashMap<>();
      final List<RuleApplication> appliedRules = new ArrayList<>();
      if(ua instanceof UnitApplicationVariable) {
         params = MomotUtil.extractParameters(null, ((UnitApplicationVariable) ua).getAssignment());
         appliedRules.addAll(((UnitApplicationVariable) ua).getAppliedRules());
      } else if(ua instanceof RuleApplicationVariable) {
         final RuleApplicationVariable[] rarr = { (RuleApplicationVariable) ua };
         params = MomotUtil.extractParameters(null, rarr);
      }

      final IApplicationState a = new ApplicationState(ua.getUnit(), -1, params, appliedRules);
      return a;
   }

   @Override
   public void writeQTableToDisk(final IQTableAccessor<List<IApplicationState>, List<IApplicationState>> qTable,
         final String outputSrc) {

      FileOutputStream fos = null;
      ObjectOutputStream out = null;
      try {
         fos = new FileOutputStream(outputSrc);
         out = new ObjectOutputStream(fos);
         out.writeObject(qTable);
         out.close();
      } catch(final IOException ex) {
         ex.printStackTrace();
      }

   }

}
