package at.ac.tuwien.big.momot.search.criterion;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.moeaframework.core.Solution;
import org.moeaframework.core.TerminationCondition;

public abstract class ThresholdCondition implements TerminationCondition {

   protected Map<Integer, Double> objectiveThresholds;
   protected Solution terminationSolution;

   protected ThresholdCondition(final Map<Integer, Double> objectiveThresholds) {
      this.objectiveThresholds = objectiveThresholds;
   }

   public double[] getObjectiveThresholds() {
      final double[] v = new double[objectiveThresholds.values().size()];
      int i = 0;
      for(final double n : objectiveThresholds.values()) {
         v[i++] = n;
      }
      return v;
   }

   public Solution getTerminationSolution() {
      return terminationSolution;
   }

   public Map<Integer, Double> getThresholds() {
      return this.objectiveThresholds;
   }

   public boolean satisfiesCriteria(final Solution s) {
      final double[] o = s.getObjectives();
      for(final Entry<Integer, Double> e : objectiveThresholds.entrySet()) {

         if(o[e.getKey()] >= e.getValue()) {
            return false;
         }
      }
      return true;
   }

   public void setThreshold(final int conditionIndex, final double val) {
      this.objectiveThresholds = new HashMap<>(this.objectiveThresholds);
      this.objectiveThresholds.put(conditionIndex, val);
   }

   @Override
   public abstract String toString();
}
