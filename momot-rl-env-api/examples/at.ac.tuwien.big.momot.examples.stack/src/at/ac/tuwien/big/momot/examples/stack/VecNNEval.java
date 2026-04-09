// package at.ac.tuwien.big.momot.examples.stack;
//
// import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
// import at.ac.tuwien.big.momot.examples.stack.stack.StackPackage;
// import at.ac.tuwien.big.momot.learn.NNDelegateEval;
// import at.ac.tuwien.big.momot.learn.NNDelegateVec;
// import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
// import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies.StackEncodingStrategy;
// import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.BaseEnvironment;
// import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
// import at.ac.tuwien.big.momot.util.MomotUtil;
//
// import java.io.File;
// import java.io.IOException;
// import java.nio.file.Path;
// import java.util.ArrayList;
// import java.util.List;
// import java.util.stream.Collectors;
//
// import org.eclipse.emf.ecore.resource.Resource;
// import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
//
// import co.gongzh.procbridge.Server;
//
// public class VecNNEval {
// private static final String MODELS_PATH = "model";
//
// private static final String INPUT_MODEL = "model_twentyfive_stacks.xmi";
// private static final List<String> TEST_MODELS = List.of("model_five_stacks.xmi", "model_ten_stacks.xmi",
// "model_fifty_stacks.xmi");
// private static final List<Integer> TEST_STEPS = List.of(8, 20, 100);
//
// // private static final String INPUT_MODEL = "comparison_11-4-2-5-0-19 12-14-7-2.xmi";
// // private static final String REFERENCE_SET = "model/input/referenceSet/model_five_stacks_reference_set.pf";
//
// public static void main(final String[] args) throws IOException {
// StackPackage.eINSTANCE.eClass();
// IEGraphMultiDimensionalFitnessFunction fitnessFunction = null;
//
// // final File[] files = new File(TEST_MODELS).listFiles();
// // final List<File> fileList = Arrays.stream(files).filter(f -> !f.isDirectory()).collect(Collectors.toList());
// // final File trainFile = new File(Path.of(MODELS_PATH, INPUT_MODEL).toString());
// final HenshinResourceSet henshinResourceSet = new HenshinResourceSet();
// final Resource modelResource = henshinResourceSet.getResource(Path.of(MODELS_PATH, INPUT_MODEL).toString());
// final List<TransformationSolution> initialSolutions = new ArrayList<>();
// // final BaseEnvironment<TransformationSolution> env = learnEnv.init(search);
// final List<BaseEnvironment<TransformationSolution>> envs = new ArrayList<>();
// for(int i = 0; i < 8; i++) {
// final TransformationSearchOrchestration tso = new StackOrchestration(INPUT_MODEL, 1);
//
// // final Engine engine = new MomotEngine(true, new String());
// // final ModuleManager moduleManager = search.getModuleManager();
// // final IEGraphMultiDimensionalFitnessFunction fitnessFunction = new EGraphMultiDimensionalFitnessFunction();
// fitnessFunction = tso.getFitnessFunction();
//
// final BaseEnvironment<TransformationSolution> env = MomotUtil.initBaseLearnEnv(tso, "Standard Deviation", null,
// StackEncodingStrategy.class);
// envs.add(env);
//
// initialSolutions.add(MomotUtil.readInitialSolution(MomotUtil.eGraphOf(modelResource, true),
// fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size()));
// }
//
// final TransformationSearchOrchestration evalTso = new StackOrchestration(INPUT_MODEL, 1);
//
// final BaseEnvironment<TransformationSolution> evalEnv = MomotUtil.initBaseLearnEnv(evalTso, "Standard Deviation",
// null, StackEncodingStrategy.class);
//
// final List<File> testFiles = TEST_MODELS.stream().map(m -> new File(Path.of(MODELS_PATH, m).toString()))
// .collect(Collectors.toList());
// final List<File> fileList = testFiles.stream().filter(f -> !f.isDirectory()).collect(Collectors.toList());
//
// final List<TransformationSolution> testSolutions = MomotUtil.readInitialSolutions(fileList,
// fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size());
//
// final String outPath = "SLB-25-b144-lstm";
//
// // final Server server = new StandalonePy(env, initialSolutions, fileList);
// final Server server = new Server(8000, new NNDelegateVec(envs, outPath, fitnessFunction, initialSolutions, 50));
// final Server evalServer = new Server(8001,
// new NNDelegateEval(evalEnv, outPath, fitnessFunction, testSolutions, fileList, TEST_STEPS));
// server.start();
// evalServer.start();
//
// System.out.println("Test Server is on 8000...");
// }
//
// }
