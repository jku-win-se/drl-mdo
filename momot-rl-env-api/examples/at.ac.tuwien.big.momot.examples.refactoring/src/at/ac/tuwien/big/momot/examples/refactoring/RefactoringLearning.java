package at.ac.tuwien.big.momot.examples.refactoring;

import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.examples.refactoring.refactoring.RefactoringPackage;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies.RefactoringEncodingStrategy;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.BaseEnvironment;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.eclipse.ocl.ParserException;

public class RefactoringLearning {
   private static final String TEST_MODELS = "model/nn";

   public static void main(final String[] args) throws IOException {
      RefactoringPackage.eINSTANCE.getClass();

      // search orchestration
      final RefactoringSearch rs = new RefactoringSearch();
      TransformationSearchOrchestration tso = null;
      try {
         tso = RefactoringSearch.createOrchestration();
      } catch(final ParserException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }

      // final Engine engine = new MomotEngine(true, new String());
      // final ModuleManager moduleManager = search.getModuleManager();
      // final IEGraphMultiDimensionalFitnessFunction fitnessFunction = new EGraphMultiDimensionalFitnessFunction();
      final IEGraphMultiDimensionalFitnessFunction fitnessFunction = tso.getFitnessFunction();

      final File[] files = new File(TEST_MODELS).listFiles();
      final List<File> fileList = Arrays.stream(files).filter(f -> !f.isDirectory()).collect(Collectors.toList());

      final List<TransformationSolution> initialSolutions = MomotUtil.readInitialSolutions(fileList,
            fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size());

      // final BaseEnvironment<TransformationSolution> env = learnEnv.init(search);
      final BaseEnvironment<TransformationSolution> env = MomotUtil.initBaseLearnEnv(tso, "ContentSize", null,
            RefactoringEncodingStrategy.class);

      // final Server server = new StandalonePy(env, initialSolutions, fileList);
      // final Server server = new Server(8000, new NNDelegate(env, initialSolutions, fileList, 10));
      // server.start();
      System.out.println("Test Server is on 8000...");
   }
}
