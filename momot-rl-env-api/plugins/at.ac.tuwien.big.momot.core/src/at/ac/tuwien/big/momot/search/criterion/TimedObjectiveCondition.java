package at.ac.tuwien.big.momot.search.criterion;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.moeaframework.core.Algorithm;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Solution;

public class TimedObjectiveCondition extends ThresholdCondition {

   public static TimedObjectiveCondition create(final Map<Integer, Double> objectiveThresholds) {
      return new TimedObjectiveCondition(objectiveThresholds);
   }

   private long maxNanosIfObjectiveSatisfied;
   private long startTime;
   private long timeTilObjectivesSatisfied;
   private boolean areObjectivesSatisfised;
   private long terminatedAfterNanos;

   private TimedObjectiveCondition(final Map<Integer, Double> objectiveThresholds) {
      super(new HashMap<>(objectiveThresholds));
      // this.maxNanosIfObjectiveSatisfied = (long) (maxSecondsIfObjectiveSatisfied * Math.pow(10, 9));
      this.areObjectivesSatisfised = false;
      this.terminatedAfterNanos = 0;
      this.timeTilObjectivesSatisfied = 0;
      this.maxNanosIfObjectiveSatisfied = 0;
   }

   public long getMaxNanosIfObjectiveSatisfied() {
      return maxNanosIfObjectiveSatisfied;
   }

   public long getStartTime() {
      return startTime;
   }

   public long getTerminatedAfterNanos() {
      return terminatedAfterNanos;
   }

   public long getTimeTilObjectivesSatisfied() {
      return timeTilObjectivesSatisfied;
   }

   @Override
   public void initialize(final Algorithm algorithm) {
      this.startTime = System.nanoTime();
   }

   public boolean isAreObjectivesSatisfised() {
      return areObjectivesSatisfised;
   }

   public void setMaxSecondsIfObjectiveSatisfied(final long timeInNanos) {
      this.maxNanosIfObjectiveSatisfied = timeInNanos;
   }

   @Override
   public boolean shouldTerminate(final Algorithm algorithm) {

      final long elapsedTime = System.nanoTime() - this.startTime;
      final NondominatedPopulation p = algorithm.getResult();
      double minObj = Double.POSITIVE_INFINITY;
      // Solution minSol = null;
      for(final Solution s : p) {
         final double curObj = s.getObjective(0);
         if(curObj < minObj) {
            minObj = curObj;
         }
      }
      System.out.println(minObj);
      if(!areObjectivesSatisfised) {

         // double minObj = Double.POSITIVE_INFINITY;
         // for(final Solution s : p) {
         // final double curObj = s.getObjective(0);
         // if(curObj < minObj) {
         // minObj = curObj;
         // }
         // }
         // System.out.println(minObj);

         for(final Solution s : p) {
            if(satisfiesCriteria(s)) {
               this.areObjectivesSatisfised = true;
               this.timeTilObjectivesSatisfied = elapsedTime;
               System.out.println("Found after " + algorithm.getNumberOfEvaluations() + " evaluations");
               this.terminationSolution = s;
            }
         }
      }
      if(areObjectivesSatisfised && this.maxNanosIfObjectiveSatisfied <= elapsedTime) {
         this.terminatedAfterNanos = elapsedTime;
         // Solution optimalSolution = fitSolutions.stream().min((s1, s2) -> s1.getObjective(objectiveOptimalityIndex) >
         // s2.getObjective(objectiveOptimalityIndex) ? 1 : -1).get();

         System.out.println("Terminated after " + elapsedTime / Math.pow(10, 9) + "s");

         return true;
      }

      return false;
   }

   @Override
   public String toString() {
      final StringBuilder sb = new StringBuilder();
      for(final Entry<Integer, Double> entry : objectiveThresholds.entrySet()) {
         sb.append(String.format("Timed_max%.2fs_%s-%s", maxNanosIfObjectiveSatisfied / Math.pow(10, 9),
               entry.getKey().toString(), entry.getValue().toString()));
      }
      return sb.toString();
   }

}
