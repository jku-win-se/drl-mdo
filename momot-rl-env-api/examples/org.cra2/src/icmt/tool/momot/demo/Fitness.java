package icmt.tool.momot.demo;

import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies.CRAEncodingStrategy;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.BaseEnvironment;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.ArchitectureSearcher;

public class Fitness {

   private static final String INPUT_MODEL = "problem/test/Cart_Item.xmi";
   private static final int SOLUTION_LENGTH = 5;

   public static void main(final String[] args) {
      ArchitecturePackage.eINSTANCE.eClass();

      final TransformationSearchOrchestration tso = ArchitectureSearcher.createOrchestration(INPUT_MODEL,
            SOLUTION_LENGTH);

      // final Engine engine = new MomotEngine(true, new String());
      // final ModuleManager moduleManager = search.getModuleManager();
      // final IEGraphMultiDimensionalFitnessFunction fitnessFunction = new EGraphMultiDimensionalFitnessFunction();
      final IEGraphMultiDimensionalFitnessFunction fitnessFunction = tso.getFitnessFunction();

      final File[] files = new File[1];
      files[0] = new File(INPUT_MODEL);
      final List<TransformationSolution> initialSolutions = MomotUtil.readInitialSolutions(List.of(files),
            fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size());
      final TransformationSolution ts = initialSolutions.get(0);
      ts.execute();
      fitnessFunction.evaluate(ts);

      System.out.println(Arrays.toString(ts.getObjectives()));

      // final BaseEnvironment<TransformationSolution> env = learnEnv.init(search);
      final BaseEnvironment<TransformationSolution> env = MomotUtil.initBaseLearnEnv(tso, "CRAIndex", null,
            CRAEncodingStrategy.class);

   }
}
