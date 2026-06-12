package at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures;

import java.util.List;
import java.util.Map;

import org.eclipse.emf.henshin.interpreter.RuleApplication;
import org.eclipse.emf.henshin.model.Unit;

public interface IApplicationState {
   public List<RuleApplication> getAppliedRules();

   public Map<String, Object> getParamValues();

   public Unit getUnit();

   public String getUnitName();

   public void setUnit(final Unit unit);
}
