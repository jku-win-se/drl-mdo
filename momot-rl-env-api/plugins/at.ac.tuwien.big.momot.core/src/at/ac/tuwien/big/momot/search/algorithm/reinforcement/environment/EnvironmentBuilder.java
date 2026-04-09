package at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IRewardStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.IEnvironment;

import java.util.HashMap;
import java.util.Map;
import java.util.function.ToDoubleFunction;

import org.eclipse.emf.ecore.EObject;
import org.moeaframework.core.Solution;

public class EnvironmentBuilder<S extends Solution> {
   private IRewardStrategy<S> rewardStrategy;

   private IEncodingStrategy<S> encodingStrategy;
   // private final IEGraphMultiDimensionalFitnessFunction fitnessFunction;
   // private String objectiveName;
   // private String constraintName;
   // private IEGraphMultiDimensionalFitnessFunction realFitnessFunction;

   private ToDoubleFunction<double[]> fitnessAggregator;
   private ToDoubleFunction<double[]> constraintAggregator;
   ToDoubleFunction<EObject> fitnessScaler;
   ToDoubleFunction<EObject> constraintScaler;

   public EnvironmentBuilder() {
      // this.fitnessFunction = fitnessFunction;
      this.rewardStrategy = null;
      this.encodingStrategy = null;
      // this.objectiveName = null;
      // this.constraintName = null;
      this.fitnessAggregator = null;
      this.constraintAggregator = null;
      this.fitnessScaler = null;
      this.constraintScaler = null;
      // this.realFitnessFunction = null;
   }

   @SuppressWarnings("unchecked")
   public Map<IEnvironment.Type, IEnvironment<S>> build() {
      final Map<IEnvironment.Type, IEnvironment<S>> environmentMap = new HashMap<>();

      environmentMap.put(IEnvironment.Type.SOValueBasedEnvironment,
            new SOEnvironment<>(this.fitnessAggregator, this.constraintAggregator, this.fitnessScaler,
                  this.constraintScaler, this.rewardStrategy, this.encodingStrategy));

      // final List<IFitnessComparator<?, Solution>> comparatorList = fitnessFunction.getObjectiveNames().stream()
      // .map(x -> fitnessFunction.getObjectiveIndex(x)).map(y -> new ObjectiveFitnessComparator<>(y))
      // .collect(Collectors.toList());
      //
      // environmentMap.put(IEnvironment.Type.MOValueBasedEnvironment, new MOEnvironment<>(fitnessFunction,
      // comparatorList,
      // fitnessFunction.getObjectiveNames(), this.encodingStrategy));

      return environmentMap;

   }

   public EnvironmentBuilder<S> constraintAggregator(final ToDoubleFunction<double[]> constraintAggregator) {
      this.constraintAggregator = constraintAggregator;
      return this;
   }

   public EnvironmentBuilder<S> constraintScaler(final ToDoubleFunction<EObject> constraintScaler) {
      this.constraintScaler = constraintScaler;
      return this;
   }

   public EnvironmentBuilder<S> encodingStrategy(final IEncodingStrategy<S> strategy) {
      this.encodingStrategy = strategy;
      return this;
   }
   //
   // public EnvironmentBuilder<S> realFitnessFunction(final IEGraphMultiDimensionalFitnessFunction fitnessFunction) {
   // this.realFitnessFunction = fitnessFunction;
   // return this;
   // }

   public EnvironmentBuilder<S> fitnessAggregator(final ToDoubleFunction<double[]> fitnessAggregator) {
      this.fitnessAggregator = fitnessAggregator;
      return this;
   }

   public EnvironmentBuilder<S> fitnessScaler(final ToDoubleFunction<EObject> fitnessScaler) {
      this.fitnessScaler = fitnessScaler;
      return this;
   }

   public EnvironmentBuilder<S> rewardStrategy(final IRewardStrategy<S> strategy) {
      this.rewardStrategy = strategy;
      return this;
   }

   // public EnvironmentBuilder<S> singleConstraint(final String constraintName) {
   // this.constraintName = constraintName;
   // return this;
   // }
   //
   // public EnvironmentBuilder<S> singleObjective(final String objectiveName) {
   // this.objectiveName = objectiveName;
   // return this;
   // }

}
