package at.ac.tuwien.big.momot.space.exploration;

import at.ac.tuwien.big.momot.ds.ApplicationState;
import at.ac.tuwien.big.momot.problem.solution.variable.ITransformationVariable;
import at.ac.tuwien.big.momot.problem.solution.variable.UnitApplicationVariable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.henshin.interpreter.Assignment;
import org.eclipse.emf.henshin.interpreter.RuleApplication;
import org.eclipse.emf.henshin.interpreter.UnitApplication;
import org.eclipse.emf.henshin.model.Parameter;

public class ExploreUtils {
   public static ITransformationVariable appStateToVar(final ApplicationState curAppState) {
      // TODO Auto-generated method stub
      return null;
   }

   public static Map<String, Object> extractParameters(final Assignment assignment) {
      final Map<String, Object> paramValues = new HashMap<>();

      for(final Parameter p : assignment.getUnit().getParameters()) {
         if(assignment.getParameterValue(p) != null) {
            paramValues.put(p.getName(), assignment.getParameterValue(p));
         }
      }

      return paramValues;
   }

   public static Map<String, Object> extractParameters(final RuleApplication... raps) {
      final Map<String, Object> paramValues = new HashMap<>();

      for(final RuleApplication ra : raps) {
         for(final Parameter p : ra.getUnit().getParameters()) {
            if(ra.getResultMatch().getParameterValue(p) != null) {
               paramValues.put(p.getName(), ra.getResultMatch().getParameterValue(p));
            }
         }

      }
      return paramValues;
   }

   public static ApplicationState unitApplicationToApplicationState(final UnitApplication ua) {
      Map<String, Object> params = new HashMap<>();
      final List<RuleApplication> appliedRules = new ArrayList<>();
      if(ua instanceof RuleApplication) {
         final RuleApplication[] rarr = { (RuleApplication) ua };
         params = extractParameters(rarr);
      } else if(ua instanceof UnitApplication) {
         params = extractParameters(ua.getAssignment());
         appliedRules.addAll(((UnitApplicationVariable) ua).getAppliedRules());
      }

      final ApplicationState a = new ApplicationState(ua.getUnit(), params, appliedRules);
      return a;
   }

}
