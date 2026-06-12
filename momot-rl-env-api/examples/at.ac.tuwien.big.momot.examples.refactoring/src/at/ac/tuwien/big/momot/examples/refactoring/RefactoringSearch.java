package at.ac.tuwien.big.momot.examples.refactoring;

import at.ac.tuwien.big.moea.SearchAnalysis;
import at.ac.tuwien.big.moea.SearchExperiment;
import at.ac.tuwien.big.moea.SearchResultManager;
import at.ac.tuwien.big.moea.experiment.analyzer.SearchAnalyzer;
import at.ac.tuwien.big.moea.experiment.executor.SearchExecutor;
import at.ac.tuwien.big.moea.experiment.executor.listener.CurrentNondominatedPopulationPrintListener;
import at.ac.tuwien.big.moea.experiment.executor.listener.SeedRuntimePrintListener;
import at.ac.tuwien.big.moea.search.algorithm.EvolutionaryAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.LocalSearchAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.provider.IRegisteredAlgorithm;
import at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension;
import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.TransformationResultManager;
import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.examples.refactoring.refactoring.RefactoringModel;
import at.ac.tuwien.big.momot.examples.refactoring.refactoring.RefactoringPackage;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.operator.mutation.TransformationPlaceholderMutation;
import at.ac.tuwien.big.momot.search.algorithm.operator.mutation.TransformationVariableMutation;
import at.ac.tuwien.big.momot.search.fitness.EGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.dimension.AbstractEGraphFitnessDimension;
import at.ac.tuwien.big.momot.search.fitness.dimension.OCLQueryDimension;
import at.ac.tuwien.big.momot.search.fitness.dimension.TransformationLengthDimension;
import at.ac.tuwien.big.momot.search.solution.repair.ITransformationRepairer;
import at.ac.tuwien.big.momot.search.solution.repair.TransformationPlaceholderRepairer;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.ocl.ParserException;
import org.moeaframework.algorithm.NSGAII;
import org.moeaframework.core.Algorithm;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.operator.OnePointCrossover;
import org.moeaframework.core.operator.TournamentSelection;
import org.moeaframework.util.progress.ProgressListener;

public class RefactoringSearch {

   protected static String CASE = "SeveralRefactorings";

   private static String INPUT_MODEL = "model/" + CASE + ".xmi";
   protected static int SOLUTION_LENGTH = 10;

   protected static boolean PRINT_POPULATIONS = false;
   protected static String PRINT_DIRECTORY = "output/populations/SeveralRefactorings";

   protected static int MAX_EVALUATIONS = 1000;

   protected static int NR_RUNS = 1;

   public static TransformationSearchOrchestration createOrchestration() throws ParserException {
      final ModuleManager henshin = new ModuleManager("model/", "Refactoring.henshin");
      final TransformationSearchOrchestration orchestration = new TransformationSearchOrchestration(henshin,
            "SeveralRefactorings.xmi", 10);

      final IEGraphMultiDimensionalFitnessFunction fitnessFunction = orchestration.getFitnessFunction();

      // fitnessFunction.addObjective(new OCLQueryDimension("ContentSize", "properties->size() * 1.1 +
      // entities->size()",
      // orchestration.createOCLHelper()));

      fitnessFunction.addObjective(new AbstractEGraphFitnessDimension("ContentSize") {
         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final RefactoringModel rm = MomotUtil.getRoot(solution.execute(), RefactoringModel.class);
            return rm.getProperties().size() + rm.getEntities().size();
         }
      });

      fitnessFunction.addObjective(new TransformationLengthDimension());
      fitnessFunction.setSolutionRepairer(new TransformationPlaceholderRepairer());
      orchestration.setSolutionWriter(new RefactoringSolutionWriter(orchestration.getFitnessFunction()));

      return orchestration;
   }

   public static void initialization() {
      RefactoringPackage.eINSTANCE.eClass();
   }

   public static void main(final String... args) {
      initialization();
      overridefromCMDArguments(args);

      final RefactoringSearch search = new RefactoringSearch();
      search.performSearch(INPUT_MODEL, SOLUTION_LENGTH);
   }

   private static void overridefromCMDArguments(final String[] args) {
      final Options options = new Options();
      options.addOption("m", "model", true, "Case ID to run");
      options.addOption("s", "solution_length", true, "");
      options.addOption("r", "runs", true, "");
      options.addOption("e", "evaluations", true, "");

      final CommandLineParser parser = new DefaultParser();
      CommandLine cmd = null;
      try {
         cmd = parser.parse(options, args);
      } catch(final ParseException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }

      if(cmd == null || !cmd.hasOption("model")) {
         System.out.println("Running without cmd arguments...");
      } else {
         CASE = String.valueOf(cmd.getOptionValue("model"));
         SOLUTION_LENGTH = Integer.parseInt(cmd.getOptionValue("solution_length"));
         INPUT_MODEL = "model/" + CASE + ".xmi";
         PRINT_DIRECTORY = Paths.get(".", "output", "populations", CASE).toString();
         NR_RUNS = Integer.parseInt(cmd.getOptionValue("runs"));
         MAX_EVALUATIONS = Integer.parseInt(cmd.getOptionValue("evaluations"));
      }

   }

   protected final int POPULATION_SIZE = 15;

   protected final String[] modules = new String[] { "transformations/Refactoring.henshin" };

   protected final ITransformationRepairer solutionRepairer = new TransformationPlaceholderRepairer();

   protected String baseName;

   protected double significanceLevel = 0.01;

   protected ProgressListener _createListener_0() {
      final SeedRuntimePrintListener _seedRuntimePrintListener = new SeedRuntimePrintListener();
      return _seedRuntimePrintListener;
   }

   protected IFitnessDimension<TransformationSolution> _createObjective_0(
         final TransformationSearchOrchestration orchestration) {
      final IFitnessDimension<TransformationSolution> dimension = _createObjectiveHelper_0();
      dimension.setName("SolutionLength");
      dimension.setFunctionType(at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum);
      return dimension;
   }

   protected IFitnessDimension<TransformationSolution> _createObjective_1(
         final TransformationSearchOrchestration orchestration) {
      try {
         return new OCLQueryDimension("ContentSize",
               at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum,
               "properties->size() + entities->size()", orchestration.createOCLHelper());
      } catch(final ParserException e) {
         e.printStackTrace();
      }
      return null;
   }

   protected IFitnessDimension<TransformationSolution> _createObjectiveHelper_0() {
      final TransformationLengthDimension _transformationLengthDimension = new TransformationLengthDimension();
      return _transformationLengthDimension;
   }

   protected IRegisteredAlgorithm<NSGAII> _createRegisteredAlgorithm_0(
         final TransformationSearchOrchestration orchestration,
         final EvolutionaryAlgorithmFactory<TransformationSolution> moea,
         final LocalSearchAlgorithmFactory<TransformationSolution> local) {
      final TournamentSelection _tournamentSelection = new TournamentSelection(2);
      final OnePointCrossover _onePointCrossover = new OnePointCrossover(1.0);
      final TransformationPlaceholderMutation _transformationPlaceholderMutation = new TransformationPlaceholderMutation(
            0.15);
      final IRegisteredAlgorithm<NSGAII> _createNSGAII = moea.createNSGAII(_tournamentSelection, _onePointCrossover,
            _transformationPlaceholderMutation);
      return _createNSGAII;
   }

   protected IRegisteredAlgorithm<NSGAII> _createRegisteredAlgorithm_1(
         final TransformationSearchOrchestration orchestration,
         final EvolutionaryAlgorithmFactory<TransformationSolution> moea,
         final LocalSearchAlgorithmFactory<TransformationSolution> local) {
      final TournamentSelection _tournamentSelection = new TournamentSelection(2);
      final OnePointCrossover _onePointCrossover = new OnePointCrossover(1.0);
      final TransformationPlaceholderMutation _transformationPlaceholderMutation = new TransformationPlaceholderMutation(
            0.15);
      final IRegisteredAlgorithm<NSGAII> _createNSGAIII = moea.createNSGAIII(_tournamentSelection, _onePointCrossover,
            _transformationPlaceholderMutation);
      return _createNSGAIII;
   }

   protected SearchExperiment<TransformationSolution> createExperiment(
         final TransformationSearchOrchestration orchestration) {
      final SearchExperiment<TransformationSolution> experiment = new SearchExperiment<>(orchestration,
            MAX_EVALUATIONS);
      experiment.setNumberOfRuns(NR_RUNS);
      experiment.addProgressListener(_createListener_0());

      if(PRINT_POPULATIONS) {
         final List<String> algoNames = new ArrayList<>();
         for(final IRegisteredAlgorithm<? extends Algorithm> a : orchestration.getAlgorithms()) {
            algoNames.add(orchestration.getAlgorithmName(a));
         }

         if(!Files.exists(Paths.get(PRINT_DIRECTORY))) {
            new File(PRINT_DIRECTORY).mkdirs();
         }

         experiment.addProgressListener(new CurrentNondominatedPopulationPrintListener(PRINT_DIRECTORY, algoNames,
               NR_RUNS, 100, MAX_EVALUATIONS));
      }

      return experiment;
   }

   protected IEGraphMultiDimensionalFitnessFunction createFitnessFunction(
         final TransformationSearchOrchestration orchestration) {
      final IEGraphMultiDimensionalFitnessFunction function = new EGraphMultiDimensionalFitnessFunction();
      function.addObjective(_createObjective_1(orchestration));
      function.addObjective(_createObjective_0(orchestration));
      function.setSolutionRepairer(solutionRepairer);
      return function;
   }

   protected EGraph createInputGraph(final String initialGraph, final ModuleManager moduleManager) {
      final EGraph graph = moduleManager.loadGraph(initialGraph);
      return graph;
   }

   protected ModuleManager createModuleManager() {
      final ModuleManager manager = new ModuleManager();
      manager.addModules(modules);
      return manager;
   }

   protected TransformationSearchOrchestration createOrchestration(final String initialGraph,
         final int solutionLength) {
      final TransformationSearchOrchestration orchestration = new TransformationSearchOrchestration();
      final ModuleManager moduleManager = createModuleManager();
      final EGraph graph = createInputGraph(initialGraph, moduleManager);
      orchestration.setModuleManager(moduleManager);
      orchestration.setProblemGraph(graph);
      orchestration.setSolutionLength(solutionLength);
      orchestration.setFitnessFunction(createFitnessFunction(orchestration));

      final EvolutionaryAlgorithmFactory<TransformationSolution> moea = orchestration
            .createEvolutionaryAlgorithmFactory(POPULATION_SIZE);

      // final EnvironmentBuilder<TransformationSolution> envBuilder = new EnvironmentBuilder<>(
      // orchestration.getFitnessFunction());
      //
      // final Map<IEnvironment.Type, IEnvironment<TransformationSolution>> env =
      // envBuilder.singleObjective("ContentSize")
      // .build();

      // final RLAlgorithmFactory<TransformationSolution> rl = orchestration.createRLAlgorithmFactory(env);

      // algorithm settings

      orchestration.addAlgorithm("NSGAII-2", moea.createNSGAII(new TournamentSelection(2), new OnePointCrossover(1.0),
            new TransformationVariableMutation(orchestration.getSearchHelper(), 0.05)));

      // orchestration.addAlgorithm("QLearningExplore", rl.createSingleObjectiveExploreQLearner(10, 0.9, 1.0, true,
      // (1.0 - .05) / MAX_EVALUATIONS, 0.05, "QExplore-pullup-5", 10, 0, null, null, false));

      // orchestration.addAlgorithm("QLearning", rl.createSingleObjectiveQLearner(0.9, 1.0, true,
      // (1.0 - .05) / MAX_EVALUATIONS, 0.05, "Q-pullup-5", 10, 0, null, null, false));

      return orchestration;
   }

   protected void deriveBaseName(final TransformationSearchOrchestration orchestration) {
      final EObject root = MomotUtil.getRoot(orchestration.getProblemGraph());
      if(root == null || root.eResource() == null || root.eResource().getURI() == null) {
         baseName = getClass().getSimpleName();
      } else {
         baseName = root.eResource().getURI().trimFileExtension().lastSegment();
      }
   }

   protected TransformationResultManager handleResults(final SearchExperiment<TransformationSolution> experiment) {

      final TransformationResultManager resultManager = new TransformationResultManager(experiment);

      resultManager.saveParetosetPerRun(String.valueOf(MAX_EVALUATIONS), "output/populations/" + CASE);

      System.out.println("REFERENCE SET:");
      System.out.println(SearchResultManager.printObjectives(SearchResultManager.getReferenceSet(experiment, null)));
      System.out.println(SearchResultManager.printObjectives(resultManager.createApproximationSet()));

      Population population;
      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save objectives of all algorithms to 'output/objectives/objective_values.txt'");
      SearchResultManager.saveObjectives("output/objectives/objective_values.txt", population);

      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save models of all algorithms to 'output/models/'");
      saveModels(TransformationResultManager.saveModels("output/models/", baseName, population));

      System.out.println("- Save objectives of algorithms seperately to 'output/objectives/<algorithm>.txt'");
      System.out.println("- Save models of algorithms seperately to 'output/solutions/<algorithm>.txt'´\n");

      for(final Entry<SearchExecutor, List<NondominatedPopulation>> entry : resultManager.getResults().entrySet()) {

         System.out.println(entry.getKey().getName());

         population = SearchResultManager.createApproximationSet(experiment, entry.getKey().getName());
         System.out.println(SearchResultManager.printObjectives(population) + "\n");
         saveModels(TransformationResultManager.saveModels("output/models/" + entry.getKey().getName(),
               entry.getKey().getName(), population));

         population = SearchResultManager.createApproximationSet(experiment, entry.getKey().getName());
         SearchResultManager.saveObjectives("output/objectives/" + entry.getKey().getName() + ".txt", population);
      }

      return resultManager;
   }

   protected SearchAnalyzer performAnalysis(final SearchExperiment<TransformationSolution> experiment) {
      final SearchAnalysis analysis = new SearchAnalysis(experiment);
      analysis.setHypervolume(true);
      analysis.setShowAggregate(true);
      analysis.setShowIndividualValues(true);
      analysis.setShowStatisticalSignificance(true);
      analysis.setSignificanceLevel(significanceLevel);
      final SearchAnalyzer searchAnalyzer = analysis.analyze();
      System.out.println("---------------------------");
      System.out.println("Analysis Results");
      System.out.println("---------------------------");
      searchAnalyzer.printAnalysis();
      System.out.println("---------------------------");
      try {
         System.out.println("- Save Analysis to 'output/analysis/analysis.txt'");
         searchAnalyzer.saveAnalysis(new File("output/analysis/analysis.txt"));
      } catch(final IOException e) {
         e.printStackTrace();
      }
      System.out.println("- Save Indicator BoxPlots to 'output/analysis/'");
      searchAnalyzer.saveIndicatorBoxPlots("output/analysis/", baseName);
      return searchAnalyzer;
   }

   public void performSearch(final String initialGraph, final int solutionLength) {
      final TransformationSearchOrchestration orchestration = createOrchestration(initialGraph, solutionLength);
      deriveBaseName(orchestration);
      printSearchInfo(orchestration);
      final SearchExperiment<TransformationSolution> experiment = createExperiment(orchestration);
      experiment.run();
      System.out.println("-------------------------------------------------------");
      System.out.println("Analysis");
      System.out.println("-------------------------------------------------------");
      performAnalysis(experiment);
      System.out.println("-------------------------------------------------------");
      System.out.println("Results");
      System.out.println("-------------------------------------------------------");
      handleResults(experiment);

   }

   public void printSearchInfo(final TransformationSearchOrchestration orchestration) {
      System.out.println("-------------------------------------------------------");
      System.out.println("Search");
      System.out.println("-------------------------------------------------------");
      System.out.println("InputModel:      " + INPUT_MODEL);
      System.out.println("Objectives:      " + orchestration.getFitnessFunction().getObjectiveNames());
      System.out.println("NrObjectives:    " + orchestration.getNumberOfObjectives());
      System.out.println("Constraints:     " + orchestration.getFitnessFunction().getConstraintNames());
      System.out.println("NrConstraints:   " + orchestration.getNumberOfConstraints());
      System.out.println("Transformations: " + Arrays.toString(modules));
      System.out.println("Units:           " + orchestration.getModuleManager().getUnits());
      System.out.println("SolutionLength:  " + orchestration.getSolutionLength());
      System.out.println("PopulationSize:  " + POPULATION_SIZE);
      System.out.println("Iterations:      " + MAX_EVALUATIONS / POPULATION_SIZE);
      System.out.println("MaxEvaluations:  " + MAX_EVALUATIONS);
      System.out.println("AlgorithmRuns:   " + NR_RUNS);
      System.out.println("---------------------------");
   }

   public void saveModels(final List<File> modelFiles) {
      for(final File file : modelFiles) {
         final EGraph graph = MomotUtil.loadGraph(file.getPath());
         MomotUtil.saveGraph(graph, file.getPath());
      }
   }
}
