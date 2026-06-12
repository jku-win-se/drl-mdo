package at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.FixedRuleApplicationStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.UnitParameter;
import at.ac.tuwien.big.momot.examples.refactoring.refactoring.impl.RefactoringModelImpl;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.AbstractEncodingStrategy;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.util.List;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.moeaframework.core.Solution;
import org.nd4j.linalg.api.ndarray.INDArray;

public class RefactoringEncodingStrategy<S extends Solution> extends AbstractEncodingStrategy<S> {

   @Override
   public List<UnitParameter> createBaseRules() {
      throw new RuntimeException("Not implemented");
   }

   @Override
   public List<UnitParameter> createPostBaseRules() {
      throw new RuntimeException("Not implemented");

   }

   @Override
   public INDArray[] encodeSolution(final S s) {
      throw new RuntimeException("Not implemented");

   }

   @Override
   public int getActionSpace(final S initialSolution) {
      throw new RuntimeException("Not implemented");
   }

   @Override
   public ComputationGraph getActorArchitecture(final S solution, final double lr, final double l2Regularization,
         final double dropoutRate, final boolean printArchitecture) {
      throw new RuntimeException("Not implemented");

   }

   @Override
   public ComputationGraph getCriticArchitecture(final S solution, final double lr, final double l2Regularization,
         final double dropoutRate, final boolean printArchitecture) {
      throw new RuntimeException("Not implemented");

   }

   @Override
   public ComputationGraph getDDQNArchitecture(final S solution, final double learningRate,
         final double l2Regularization, final double dropoutRate, final boolean verbose) {
      throw new RuntimeException("Not implemented");

   }

   @Override
   public String getDistrActionProbabilities(final INDArray dist, final S solution,
         final boolean renormalizeIllegalActions) {
      throw new RuntimeException("Not implemented");

   }

   @Override
   public List<String> getEpisodeEndingRules() {
      return null;
   }

   @Override
   public FixedRuleApplicationStrategy getFixedUnitApplicationStrategy(final S s, final int action) {
      throw new RuntimeException("Not implemented");
   }

   @Override
   public List<Integer> getLegalActions(final S solution) {
      throw new RuntimeException("Not implemented");

   }

   @Override
   public String getModelVisualization(final S s) {
      throw new RuntimeException("Not implemented");

   }

   @Override
   public ComputationGraph getNetworkArchitecture(final S solution, final double lr, final double l2Regularization,
         final double dropoutRate, final boolean printArchitecture) {
      throw new RuntimeException("Not implemented");

   }

   @Override
   public int getStateSpace(final S initialSolution) {
      throw new RuntimeException("Not implemented");

   }

   @Override
   public boolean isEpisodeEndingState(final S s) {
      return false;
   }

   @Override
   public boolean isFinalModel(final S s) {
      return false;
   }

   @Override
   public String printModel(final S s) {
      final StringBuilder sb = new StringBuilder();

      final RefactoringModelImpl m = MomotUtil.getRoot(((TransformationSolution) s).getResultGraph(),
            RefactoringModelImpl.class);
      System.out.println("Entities + Properties: " + (m.getEntities().size() + m.getProperties().size()));
      return sb.toString();

      // final StringBuilder sb = new StringBuilder();
      // for(final icmt.tool.momot.demo.architecture.Class c : m.getEntities()) {
      // final String classFeaturesCsv = c.getEncapsulates().stream().map(f -> f.getName())
      // .collect(Collectors.joining(","));
      // System.out.println("Class: " + classFeaturesCsv);
      //
      // }

   }

}
