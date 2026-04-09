package at.ac.tuwien.big.momot.search.criterion;

import java.util.Map;
import java.util.Map.Entry;

import org.moeaframework.core.Algorithm;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Solution;

public class MinimumObjectiveCondition extends ThresholdCondition {

   public static MinimumObjectiveCondition create(final Map<Integer, Double> objectiveThresholds) {
      return new MinimumObjectiveCondition(objectiveThresholds);
   }

   private MinimumObjectiveCondition(final Map<Integer, Double> objectiveThresholds) {
      super(objectiveThresholds);
   }

   @Override
   public void initialize(final Algorithm algorithm) {
      // No initialization required; Early stopping only if objective threshold reached
   }

   @Override
   public boolean shouldTerminate(final Algorithm algorithm) {
      final NondominatedPopulation p = algorithm.getResult();
      // if (algorithm instanceof EvolutionaryAlgorithm) {
      // ((EvolutionaryAlgorithm) algorithm).getPopulation()
      // }
      double minObj = Double.POSITIVE_INFINITY;
      // Solution minSol = null;
      for(final Solution s : p) {
         final double curObj = s.getObjective(0);
         if(curObj < minObj) {
            minObj = curObj;
         }
      }
      System.out.println(minObj);

      for(final Solution s : p) {
         // System.out.println(s.getObjective(0));

         if(satisfiesCriteria(s)) {
            System.out.println("Finished after " + algorithm.getNumberOfEvaluations() + " evaluations");
            this.terminationSolution = s;
            return true;

         }
      }

      return false;
   }

   @Override
   public String toString() {
      final StringBuilder sb = new StringBuilder();
      for(final Entry<Integer, Double> entry : objectiveThresholds.entrySet()) {
         sb.append(String.format("_%s-%s", entry.getKey().toString(), entry.getValue().toString()));
      }
      return sb.toString();
   }

}
