// package icmt.tool.momot.demo;
//
// import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
// import at.ac.tuwien.big.momot.learn.NNDelegateEval;
// import at.ac.tuwien.big.momot.learn.NNDelegateVec;
// import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
// import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies.CRAEncodingStrategy;
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
// import icmt.tool.momot.demo.architecture.ArchitecturePackage;
// import icmt.tool.momot.demo.architecture.ArchitectureSearcher;
//
// public class VecNNEval {
//
// private static final String MODELS_PATH = "problem";
// private static final String INPUT_MODEL = "problem/CRA_D_classed.xmi";
// private static final List<String> TEST_MODELS = List.of("Cart_Item_classed.xmi", "CRA_C_classed.xmi");
//
// // private static final String INPUT_MODEL = "comparison_11-4-2-5-0-19 12-14-7-2.xmi";
// // private static final String REFERENCE_SET = "model/input/referenceSet/model_five_stacks_reference_set.pf";
//
// private static final List<Integer> TEST_STEPS = List.of(16, 70);
//
// public static void main(final String[] args) throws IOException {
// ArchitecturePackage.eINSTANCE.eClass();
//
// // final Engine engine = new MomotEngine(true, new String());
// // final ModuleManager moduleManager = search.getModuleManager();
// // final IEGraphMultiDimensionalFitnessFunction fitnessFunction = new EGraphMultiDimensionalFitnessFunction();
// IEGraphMultiDimensionalFitnessFunction fitnessFunction = null;
//
// // final File trainFile = new File(INPUT_MODEL);
// final HenshinResourceSet henshinResourceSet = new HenshinResourceSet();
// final Resource modelResource = henshinResourceSet.getResource(INPUT_MODEL);
// final List<TransformationSolution> initialSolutions = new ArrayList<>();
// // final BaseEnvironment<TransformationSolution> env = learnEnv.init(search);
// final List<BaseEnvironment<TransformationSolution>> envs = new ArrayList<>();
// for(int i = 0; i < 4; i++) {
// final TransformationSearchOrchestration tso = ArchitectureSearcher.createOrchestration(INPUT_MODEL, 5);
// fitnessFunction = tso.getFitnessFunction();
// final BaseEnvironment<TransformationSolution> env = MomotUtil.initBaseLearnEnv(tso, "CRAIndex", null,
// CRAEncodingStrategy.class);
// envs.add(env);
//
// initialSolutions.add(MomotUtil.readInitialSolution(MomotUtil.eGraphOf(modelResource, true),
// fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size()));
// }
//
// final TransformationSearchOrchestration evalTso = ArchitectureSearcher.createOrchestration(INPUT_MODEL, 5);
// final BaseEnvironment<TransformationSolution> evalEnv = MomotUtil.initBaseLearnEnv(evalTso, "CRAIndex", null,
// CRAEncodingStrategy.class);
//
// final List<File> testFiles = TEST_MODELS.stream().map(m -> new File(Path.of(MODELS_PATH, m).toString()))
// .collect(Collectors.toList());
// final List<File> fileList = testFiles.stream().filter(f -> !f.isDirectory()).collect(Collectors.toList());
//
// final List<TransformationSolution> testSolutions = MomotUtil.readInitialSolutions(fileList,
// fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size());
//
// final String outPath = "CRA_B-ar";
//
// // final Server server = new StandalonePy(env, initialSolutions, fileList);
// final Server server = new Server(8000, new NNDelegateVec(envs, outPath, fitnessFunction, initialSolutions, 36));
//
// final Server evalServer = new Server(8001,
// new NNDelegateEval(evalEnv, outPath, fitnessFunction, testSolutions, fileList, TEST_STEPS));
// server.start();
// evalServer.start();
//
// System.out.println("Test Server is on 8000...");
//
// Runtime.getRuntime().addShutdownHook(new Thread(() -> {
// System.out.println("Shutdown hook triggered. Cleaning up...");
// server.stop();
// evalServer.stop();
// }));
// }
//
// }
