package sprintplanning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

import SprintPlanning.Plan;
import SprintPlanning.Stakeholder;
import SprintPlanning.WorkItem;

public final class FitnessUtils {

   public static double avgCombined(final Plan plan) {
      // Plan plan = (Plan) solution.getModel();
      // System.out.println("tetS");
      final StandardDeviation stdDevCalculator = new StandardDeviation();
      final ArrayList<Double> stakeholderImportanceDeviation = new ArrayList<>();

      for(final Stakeholder stakeholder : plan.getStakeholders()) {
         final double[] importanceValues = plan.getSprints().stream()
               .map((sprint) -> sprint.getCommittedItem().stream()
                     .filter((item) -> item.getStakeholder().equals(stakeholder))
                     .mapToInt((item) -> item.getImportance()).sum())
               .mapToDouble((intSum) -> intSum.doubleValue()).toArray(); // List of total importance of each sprint for
                                                                         // this stakeholder (list contains zero for
                                                                         // empty sprints)
         final Double importanceDeviation = stdDevCalculator.evaluate(importanceValues); // Empty sprints increase
                                                                                         // deviation
         stakeholderImportanceDeviation.add(importanceDeviation);
      }
      final Double totalDeviation = stakeholderImportanceDeviation.stream().reduce(0d, (res, val) -> res + val);

      return -(totalDeviation / stakeholderImportanceDeviation.size())
            - MinimiseSprintEffortDeviation.computeFitness(plan); // In total, empty sprints increase the avg. //
   }

   public static double avgCombinedNorm(final Plan plan) {
      return computeNormalizedFitness(plan) + MinimiseSprintEffortDeviation.computeNormalizedEffortFitness(plan);
   }

   public static double avgCustomerImportanceDeviation(final Plan plan) {
      // Plan plan = (Plan) solution.getModel();
      // System.out.println("tetS");
      final StandardDeviation stdDevCalculator = new StandardDeviation();
      final ArrayList<Double> stakeholderImportanceDeviation = new ArrayList<>();

      for(final Stakeholder stakeholder : plan.getStakeholders()) {
         final double[] importanceValues = plan.getSprints().stream()
               .map((sprint) -> sprint.getCommittedItem().stream()
                     .filter((item) -> item.getStakeholder().equals(stakeholder))
                     .mapToInt((item) -> item.getImportance()).sum())
               .mapToDouble((intSum) -> intSum.doubleValue()).toArray(); // List of total importance of each sprint for
                                                                         // this stakeholder (list contains zero for
                                                                         // empty sprints)
         final Double importanceDeviation = stdDevCalculator.evaluate(importanceValues); // Empty sprints increase
                                                                                         // deviation
         stakeholderImportanceDeviation.add(importanceDeviation);
      }
      final Double totalDeviation = stakeholderImportanceDeviation.stream().reduce(0d, (res, val) -> res + val);
      return totalDeviation / stakeholderImportanceDeviation.size(); // In total, empty sprints increase the avg. //
                                                                     // deviation
   }

   public static double avgCustomerImportanceDeviationNormalized(final Plan plan) {
      final StandardDeviation stdDevCalculator = new StandardDeviation();

      final int MAX_SPRINTS = plan.getMaxSprints();
      final List<Double> normalizedDeviations = new ArrayList<>();

      for(final Stakeholder stakeholder : plan.getStakeholders()) {
         // Actual importance per sprint for this stakeholder
         final double[] importanceValues = plan.getSprints().stream()
               .mapToDouble(sprint -> sprint.getCommittedItem().stream()
                     .filter(item -> item.getStakeholder().equals(stakeholder)).mapToInt(item -> item.getImportance())
                     .sum())
               .toArray();

         final double actualDeviation = stdDevCalculator.evaluate(importanceValues);

         // Compute worst-case deviation mathematically
         final double totalImportance = Arrays.stream(importanceValues).sum();
         final double maxPossibleDeviation;

         if(MAX_SPRINTS > 0 && totalImportance > 0) {
            final double meanImportance = totalImportance / MAX_SPRINTS;
            // Worst case: [totalImportance, 0, 0, ..., 0]
            // Variance = ((n-1) * mean^2 + (total - mean)^2) / n
            final double worstCaseVariance = (MAX_SPRINTS - 1) * Math.pow(meanImportance, 2)
                  + Math.pow(totalImportance - meanImportance, 2);
            maxPossibleDeviation = Math.sqrt(worstCaseVariance / MAX_SPRINTS);
         } else {
            maxPossibleDeviation = 0.0;
         }

         // Normalize to [0, 1]
         final double normalized = maxPossibleDeviation > 0 ? actualDeviation / maxPossibleDeviation : 0.0;
         normalizedDeviations.add(normalized);
      }

      return normalizedDeviations.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
   }

   public static double computeFitness(final Plan plan) {
      final StandardDeviation stdDevCalculator = new StandardDeviation();

      // Calculate standard deviation of importance for each stakeholder across sprints
      final double stakeholderImportanceDeviations[] = plan.getStakeholders().stream().map(stakeholder -> {
         final double[] importanceValues = plan.getSprints().stream()
               .map(sprint -> sprint.getCommittedItem().stream()
                     .filter(item -> item.getStakeholder().equals(stakeholder)).mapToInt(item -> item.getImportance())
                     .sum())
               .mapToDouble(intSum -> intSum.doubleValue()).toArray(); // Total importance per sprint for this
                                                                       // stakeholder

         return stdDevCalculator.evaluate(importanceValues); // Deviation for this stakeholder
      }).mapToDouble(Double::doubleValue).toArray();

      final double actualDeviationOfDeviations = stdDevCalculator.evaluate(stakeholderImportanceDeviations);

      return actualDeviationOfDeviations; // Lower value = more equitable distribution of variability
   }

   public static double computeNormalizedFitness(final Plan plan) {
      final StandardDeviation stdDevCalculator = new StandardDeviation();

      final int numSprints = plan.getSprints().size();

      // --- Step 1: Compute actual deviations (same as original) ---
      final double[] stakeholderImportanceDeviations = plan.getStakeholders().stream().map(stakeholder -> {
         final double[] importanceValues = plan.getSprints().stream()
               .mapToDouble(sprint -> sprint.getCommittedItem().stream()
                     .filter(item -> item.getStakeholder().equals(stakeholder)).mapToInt(item -> item.getImportance())
                     .sum())
               .toArray();

         return stdDevCalculator.evaluate(importanceValues);
      }).mapToDouble(Double::doubleValue).toArray();

      final double actualDeviationOfDeviations = stdDevCalculator.evaluate(stakeholderImportanceDeviations);

      // --- Step 2: Compute worst-case deviation-of-deviations ---
      final double[] worstCasePerStakeholder = plan.getStakeholders().stream().mapToDouble(stakeholder -> {
         final int totalImportance = plan.getSprints().stream().flatMap(s -> s.getCommittedItem().stream())
               .filter(item -> item.getStakeholder().equals(stakeholder)).mapToInt(WorkItem::getImportance).sum();

         // Worst-case stddev: all importance in one sprint, rest zeros
         return totalImportance * Math.sqrt((numSprints - 1.0) / numSprints);
      }).toArray();

      final double worstCaseDeviation = stdDevCalculator.evaluate(worstCasePerStakeholder);

      // --- Step 3: Normalize actual deviation into [0, 1] for minimization ---
      if(worstCaseDeviation == 0.0) {
         return 0.0; // Trivial case: no variation possible → already minimal
      }

      final double normalized = actualDeviationOfDeviations / worstCaseDeviation;

      // Clamp to [0, 1]
      return Math.max(0.0, Math.min(1.0, normalized));
   }

   public static double constraintViolationCombined(final Plan p) {
      // double sprintMaxViolation = 0;
      double sprintMinViolation = 0;
      double sprintMaxViolation = 0;

      final List<WorkItem> workItems = p.getBacklog().getWorkitems();
      final double noUnassignedWorkitems = workItems.stream().filter(item -> item.getIsPlannedFor().isEmpty()).count();
      final double noMultiAssignedWorkitems = workItems.stream().filter(item -> item.getIsPlannedFor().size() > 1)
            .count();

      final double currentSprints = p.getCurrentSprints();
      final double minSprints = p.getMinSprints();
      final double maxSprints = p.getMaxSprints();
      sprintMinViolation = currentSprints > minSprints ? 0 : minSprints - currentSprints;
      sprintMaxViolation = currentSprints < maxSprints ? 0 : currentSprints - maxSprints;

      return sprintMinViolation + sprintMaxViolation + noMultiAssignedWorkitems + noUnassignedWorkitems;
   }

   public static double constraintViolationNormalized(final Plan p) {
      final List<WorkItem> workItems = p.getBacklog().getWorkitems();
      final double unassignedWorkitems = workItems.stream().filter(item -> item.getIsPlannedFor().isEmpty()).count();

      final double currentSprints = p.getCurrentSprints();
      final double minSprints = p.getMinSprints();
      final double maxSprints = p.getMaxSprints();

      // Compute raw violations
      final double sprintMinViolation = currentSprints > minSprints ? 0 : minSprints - currentSprints;
      final double sprintMaxViolation = currentSprints < maxSprints ? 0 : currentSprints - maxSprints;

      // Normalize each component
      final double normalizedSprintMin = minSprints > 0 ? sprintMinViolation / minSprints : 0;
      final double normalizedSprintMax = maxSprints > 0 ? sprintMaxViolation / maxSprints : 0;
      final double normalizedUnassigned = workItems.size() > 0 ? unassignedWorkitems / workItems.size() : 0;

      // Combine and divide by 3 to ensure [0,1]
      final double combined = (normalizedSprintMin + normalizedSprintMax + normalizedUnassigned) / 3.0;

      return combined;
   }

   // public static double constraintViolationCombinedNormalized(final Plan p) {
   // // double sprintMaxViolation = 0;
   // double sprintMinViolation = 0;
   //
   // final List<WorkItem> workItems = p.getBacklog().getWorkitems();
   // final double noUnassignedWorkitems = workItems.stream().filter(item -> item.getIsPlannedFor().isEmpty()).count()
   // / (double) workItems.size();
   // final double noMultiAssignedWorkitems = workItems.stream().filter(item -> item.getIsPlannedFor().size() > 1)
   // .count() / (double) workItems.size();
   //
   // final double currentSprints = p.getCurrentSprints();
   // // final int maxSprints = p.getMaxSprints();
   // final double minSprints = p.getMinSprints();
   //
   // // sprintMaxViolation = currentSprints < maxSprints ? 0 : currentSprints - maxSprints;
   // sprintMinViolation = currentSprints > minSprints ? 0 : (minSprints - currentSprints) / minSprints;
   // // System.out.println("currentSprints - maxSprints: " + (currentSprints - maxSprints));
   //
   // return (sprintMinViolation + noMultiAssignedWorkitems + noUnassignedWorkitems) ;
   // }

   public static double noExceedingMaxNumOfSprints(final Plan p) {
      final int currentSprints = p.getCurrentSprints();
      final int maxSprints = p.getMaxSprints();
      if(currentSprints < maxSprints) {
         return 0;
      }
      // System.out.println("currentSprints - maxSprints: " + (currentSprints - maxSprints));

      return currentSprints - maxSprints;
   }

   public static double noFallingBelowMinNumOfSprints(final Plan p) {
      final int currentSprints = p.getCurrentSprints();
      final int minSprints = p.getMinSprints();
      if(currentSprints > minSprints) {
         return 0;
      }
      // System.out.println("minSprints - currentSprints: " + (minSprints - currentSprints));
      return minSprints - currentSprints;
   }

   public static double noMultiAssignedWorkItems(final Plan solution) {

      final List<WorkItem> workItems = solution.getBacklog().getWorkitems();
      // System.out.println(
      // "MultiAssignedWorkItems: " + workItems.stream().filter(item -> item.getIsPlannedFor().size() > 1).count());

      return workItems.stream().filter(item -> item.getIsPlannedFor().size() > 1).count();
   }

   public static double noUnassignedWorkitems(final Plan solution) {

      final List<WorkItem> workItems = solution.getBacklog().getWorkitems();
      // System.out.println(
      // "Unassigned workitems: " + workItems.stream().filter(item -> item.getIsPlannedFor().isEmpty()).count());
      return workItems.stream().filter(item -> item.getIsPlannedFor().isEmpty()).count(); // 0 indicates validity
   }

}
