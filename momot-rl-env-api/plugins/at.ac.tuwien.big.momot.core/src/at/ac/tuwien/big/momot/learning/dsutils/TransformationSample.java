package at.ac.tuwien.big.momot.learning.dsutils;

import at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.ApplicationState;

import com.fasterxml.jackson.annotation.JsonGetter;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(value = { "nextModelPath", "reward", "nextAppState" })
public class TransformationSample {
   public static TransformationSample of(final String modelPath, final ApplicationState as, final boolean isLast,
         final double reward, final String nextModelpath) {
      return new TransformationSample(modelPath, as, null, isLast, reward, nextModelpath);
   }

   private final String modelPath;
   private String nextModelpath;

   private final ApplicationState appState;

   @JsonIgnore
   private ApplicationState nextAppState;

   @JsonIgnore
   private boolean isLast;

   @JsonIgnore
   private double reward;

   public TransformationSample(final String modelPath, final ApplicationState applicationState,
         final ApplicationState nextAppState, final boolean isLast, final double reward, final String nextModelpath) {
      this.modelPath = modelPath;
      this.appState = applicationState;
      this.isLast = isLast;
      this.reward = reward;
      this.nextModelpath = nextModelpath;
      this.nextAppState = nextAppState;
   }

   public ApplicationState getAppState() {
      return appState;
   }

   @JsonGetter("modelPath")
   public String getModelPath() {
      return modelPath;
   }

   @JsonIgnore
   public ApplicationState getNextAppState() {
      return nextAppState;
   }

   @JsonGetter("nextModelPath")
   public String getNextModelpath() {
      return nextModelpath;
   }

   @JsonGetter("reward")
   public double getReward() {
      return reward;
   }

   @JsonGetter("isLast")
   public boolean isLast() {
      return isLast;
   }

   public void setIsLast(final boolean b) {
      this.isLast = b;
   }

   public void setNextAppState(final ApplicationState a) {
      this.nextAppState = a;
   }

   public void setNextModelpath(final String path) {
      this.nextModelpath = path;

   }

   public void setReward(final double reward) {
      this.reward = reward;

   }

}
