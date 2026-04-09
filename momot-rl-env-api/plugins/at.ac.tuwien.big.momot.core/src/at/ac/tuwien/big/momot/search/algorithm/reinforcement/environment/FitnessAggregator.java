package at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment;

public class FitnessAggregator {
   public static FitnessAggregator of(final double fitness, final double constraintViolation) {
      return new FitnessAggregator(fitness, constraintViolation);
   }

   private final double aggregatedFitness;

   private final double aggregatedConstraintViolation;

   private FitnessAggregator(final double fitness, final double constraintViolation) {
      this.aggregatedConstraintViolation = constraintViolation;
      this.aggregatedFitness = fitness;
   }

   public double getAggregatedConstraintViolation() {
      return aggregatedConstraintViolation;
   }

   public double getAggregatedFitness() {
      return aggregatedFitness;
   }

}
