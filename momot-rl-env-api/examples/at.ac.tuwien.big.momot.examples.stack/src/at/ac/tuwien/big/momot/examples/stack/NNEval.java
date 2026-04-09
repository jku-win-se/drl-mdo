// package at.ac.tuwien.big.momot.examples.stack;
//
// import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
// import at.ac.tuwien.big.momot.examples.stack.stack.StackPackage;
// import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
// import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies.StackEncodingStrategy;
// import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.BaseEnvironment;
// import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
// import at.ac.tuwien.big.momot.util.MomotUtil;
//
// import java.io.File;
// import java.io.IOException;
// import java.util.Arrays;
// import java.util.List;
// import java.util.stream.Collectors;
//
// import co.gongzh.procbridge.Server;
//
// public class NNEval {
//
// private static final String INPUT_MODEL = "model_five_stacks.xmi";
// private static final String TEST_MODELS = "model/nn";
// // private static final String INPUT_MODEL = "comparison_11-4-2-5-0-19 12-14-7-2.xmi";
//
// // private static final String REFERENCE_SET = "model/input/referenceSet/model_five_stacks_reference_set.pf";
//
// public static void main(final String[] args) throws IOException {
// StackPackage.eINSTANCE.eClass();
//
// final TransformationSearchOrchestration tso = new StackOrchestration(INPUT_MODEL, 1);
//
// // final Engine engine = new MomotEngine(true, new String());
// // final ModuleManager moduleManager = search.getModuleManager();
// // final IEGraphMultiDimensionalFitnessFunction fitnessFunction = new EGraphMultiDimensionalFitnessFunction();
// final IEGraphMultiDimensionalFitnessFunction fitnessFunction = tso.getFitnessFunction();
//
// final File[] files = new File(TEST_MODELS).listFiles();
// final List<File> fileList = Arrays.stream(files).filter(f -> !f.isDirectory()).collect(Collectors.toList());
//
// // final BaseEnvironment<TransformationSolution> env = learnEnv.init(search);
//
// final BaseEnvironment<TransformationSolution> env = MomotUtil.initBaseLearnEnv(tso, "Standard Deviation", null,
// StackEncodingStrategy.class);
//
// final TransformationSolution initialSolution = MomotUtil.readInitialSolutions(fileList,
// fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size()).get(0);
//
// // final Server server = new StandalonePy(env, initialSolutions, fileList);
// final Server server = new Server(8000, new NNDelegate(env, List.of(initialSolution), fileList, 20));
// server.start();
// System.out.println("Test Server is on 8000...");
// }
//
// }
