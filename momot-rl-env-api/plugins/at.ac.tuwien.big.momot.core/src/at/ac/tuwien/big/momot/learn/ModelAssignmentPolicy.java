package at.ac.tuwien.big.momot.learn;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

public class ModelAssignmentPolicy {
   public enum Strategy {
      ROUND_ROBIN, DETERMINISTIC_PER_ENV_AND_INSTANCE, INVERSE_PROBABILITY_BY_STEPS
   }

   private final List<Integer> stepsPerModel;
   private final int[] instanceIndexPerEnv;
   private final AtomicInteger modelIndex = new AtomicInteger(0);
   private final Random random = new Random();
   private volatile Strategy strategy = Strategy.DETERMINISTIC_PER_ENV_AND_INSTANCE;

   public ModelAssignmentPolicy(final List<Integer> stepsPerModel, final int envCount) {
      this.stepsPerModel = stepsPerModel;
      this.instanceIndexPerEnv = new int[envCount];
   }

   private int chooseModelIndex(final int envIdx, final boolean advanceInstance) {
      switch(strategy) {
         case DETERMINISTIC_PER_ENV_AND_INSTANCE:
            return getModelIndexForEnvInstance(envIdx, advanceInstance);
         case INVERSE_PROBABILITY_BY_STEPS:
            return getModelIndexInverseProbabilityBySteps();
         case ROUND_ROBIN:
         default:
            return getModelIndexRoundRobin();
      }
   }

   private int getModelIndexForEnvInstance(final int envIdx, final boolean advanceInstance) {
      return Math.floorMod(envIdx, stepsPerModel.size());

   }

   private int getModelIndexInverseProbabilityBySteps() {
      double totalWeight = 0.0;
      for(final int steps : stepsPerModel) {
         totalWeight += 1.0 / Math.max(steps, 1);
      }
      double r = random.nextDouble() * totalWeight;
      for(int i = 0; i < stepsPerModel.size(); i++) {
         r -= 1.0 / Math.max(stepsPerModel.get(i), 1);
         if(r <= 0.0) {
            return i;
         }
      }
      return stepsPerModel.size() - 1;
   }

   private int getModelIndexRoundRobin() {
      return modelIndex.getAndIncrement() % stepsPerModel.size();
   }

   public synchronized int initModelIndex(final int envIdx) {
      return chooseModelIndex(envIdx, false);
   }

   public synchronized int nextModelIndex(final int envIdx) {
      return chooseModelIndex(envIdx, true);
   }

   public synchronized void reset() {
      modelIndex.set(0);
      Arrays.fill(instanceIndexPerEnv, 0);
   }

   public void setStrategy(final Strategy strategy) {
      if(strategy != null) {
         this.strategy = strategy;
      }
   }
}
