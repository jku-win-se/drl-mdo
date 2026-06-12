package icmt.tool.momot.demo;

import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies.CRAEncodingStrategy;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.BaseEnvironment;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import co.gongzh.procbridge.Server;
import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.ArchitectureSearcher;

public class NNEval {
   private static final int SOLUTION_LENGTH = 5;
   private static final int POPULATION_SIZE = 100;
   private static final int MAX_EVALUATIONS = POPULATION_SIZE * 100;

   private static final String INPUT_MODEL = "problem/Cart_Item.xmi";
   private static final String TEST_MODELS = "problem/nn";
   // private static final String INPUT_MODEL = "comparison_11-4-2-5-0-19 12-14-7-2.xmi";

   // private static final String REFERENCE_SET = "model/input/referenceSet/model_five_stacks_reference_set.pf";

   public static void main(final String[] args) throws IOException {
      ArchitecturePackage.eINSTANCE.eClass();

      final TransformationSearchOrchestration tso = ArchitectureSearcher.createOrchestration(INPUT_MODEL,
            SOLUTION_LENGTH);

      // final Engine engine = new MomotEngine(true, new String());
      // final ModuleManager moduleManager = search.getModuleManager();
      // final IEGraphMultiDimensionalFitnessFunction fitnessFunction = new EGraphMultiDimensionalFitnessFunction();
      final IEGraphMultiDimensionalFitnessFunction fitnessFunction = tso.getFitnessFunction();

      final File[] files = new File(TEST_MODELS).listFiles();
      final List<File> fileList = Arrays.stream(files).filter(f -> !f.isDirectory()).collect(Collectors.toList());

      final List<TransformationSolution> initialSolutions = MomotUtil.readInitialSolutions(fileList,
            fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size());

      // final BaseEnvironment<TransformationSolution> env = learnEnv.init(search);
      final BaseEnvironment<TransformationSolution> env = MomotUtil.initBaseLearnEnv(tso, "CRAIndex", null,
            CRAEncodingStrategy.class);

      // final Server server = new StandalonePy(env, initialSolutions, fileList);
      final Server server = new Server(8000, new NNDelegate(env, initialSolutions, fileList, 36));
      server.start();
      System.out.println("Test Server is on 8000...");
   }

}
