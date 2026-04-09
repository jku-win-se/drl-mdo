package at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;

import org.moeaframework.core.Solution;

public class SOEnvResponse<S extends Solution> extends EnvResponse<S> {
   private double reward;

   // Sum of violations and minimization objectives
   private double aggregatedFitness;

   public double getAggregatedFitness() {
      return -this.aggregatedFitness;
   }

   public double getReward() {
      return reward;
   }

   public void setAggregatedFitness(final double aggregatedFitness) {
      this.aggregatedFitness = aggregatedFitness;
   }

   public void setReward(final double reward) {
      this.reward = reward;
   }

   public String toString(final IEncodingStrategy<S> codec) {
      return String.format("Done: %s\nAction: %d\nReward: %f\n%s", this.getDoneStatus(), this.getAppliedAction(),
            this.getReward(), codec.getModelVisualization(getState())).toString();
   }

}
