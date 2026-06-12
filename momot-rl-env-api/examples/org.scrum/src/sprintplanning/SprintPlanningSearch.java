package sprintplanning;

import at.ac.tuwien.big.moea.SearchAnalysis;
import at.ac.tuwien.big.moea.SearchExperiment;
import at.ac.tuwien.big.moea.SearchResultManager;
import at.ac.tuwien.big.moea.experiment.analyzer.SearchAnalyzer;
import at.ac.tuwien.big.moea.experiment.executor.SearchExecutor;
import at.ac.tuwien.big.moea.experiment.executor.listener.CurrentNondominatedPopulationPrintListener;
import at.ac.tuwien.big.moea.experiment.executor.listener.SeedRuntimePrintListener;
import at.ac.tuwien.big.moea.print.IPopulationWriter;
import at.ac.tuwien.big.moea.print.ISolutionWriter;
import at.ac.tuwien.big.moea.search.algorithm.EvolutionaryAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.LocalSearchAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.RLAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.provider.IRegisteredAlgorithm;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.IEnvironment;
import at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension;
import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.TransformationResultManager;
import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.operator.mutation.TransformationPlaceholderMutation;
import at.ac.tuwien.big.momot.search.algorithm.operator.mutation.TransformationVariableMutation;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.EnvironmentBuilder;
import at.ac.tuwien.big.momot.search.fitness.EGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.dimension.AbstractEGraphFitnessDimension;
import at.ac.tuwien.big.momot.search.fitness.dimension.TransformationLengthDimension;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.moeaframework.algorithm.NSGAII;
import org.moeaframework.core.Algorithm;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.operator.OnePointCrossover;
import org.moeaframework.core.operator.TournamentSelection;
import org.moeaframework.util.progress.ProgressListener;

import SprintPlanning.Plan;
import SprintPlanning.Sprint;
import SprintPlanning.SprintPlanningFactory;
import SprintPlanning.SprintPlanningPackage;
import SprintPlanning.WorkItem;

@SuppressWarnings("all")
public class SprintPlanningSearch {

   protected static String CASE = "sprint-planning-model-4-stakeholders-42-items";

   protected static String INITIAL_MODEL = "problem/" + CASE + ".xmi";

   protected static int SOLUTION_LENGTH = 100;
   // 64 - 150 // 119 - 300 // 190 - 400
   protected static int MAX_EVALUATIONS = 1000;

   protected static int NR_RUNS = 1;

   protected static boolean PRINT_POPULATIONS = false;
   protected static String PRINT_DIRECTORY = "output/populations/ScrumPlanning";

   public static IFitnessDimension<TransformationSolution> _createConstraint_rl_0() {
      return new AbstractEGraphFitnessDimension("constraintCombined",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createConstraintHelper_rl_0(solution, graph, root);
         }
      };
   }

   protected static double _createConstraintHelper_rl_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return sprintplanning.FitnessUtils.constraintViolationCombined((Plan) root);
   }

   public static IFitnessDimension<TransformationSolution> _createObjective_0() {
      return new AbstractEGraphFitnessDimension("CustomerImportanceDev",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createObjectiveHelper_0(solution, graph, root);
         }
      };
   }

   public static IFitnessDimension<TransformationSolution> _createObjective_1() {
      return new AbstractEGraphFitnessDimension("SprintEffortDev",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createObjectiveHelper_1(solution, graph, root);
         }
      };
   }

   public static IFitnessDimension<TransformationSolution> _createObjective_2() {
      final IFitnessDimension<TransformationSolution> dimension = _createObjectiveHelper_2();
      dimension.setName("SolutionLength");
      dimension.setFunctionType(at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum);
      return dimension;
   }

   public static IFitnessDimension<TransformationSolution> _createObjective_rl_0() {
      return new AbstractEGraphFitnessDimension("avgCombinedNorm",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createObjectiveHelper_rl_0(solution, graph, root);
         }
      };
   }

   protected static double _createObjectiveHelper_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.computeFitness((Plan) root);
   }

   protected static double _createObjectiveHelper_1(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return MinimiseSprintEffortDeviation.computeFitness((Plan) root);
   }

   protected static IFitnessDimension<TransformationSolution> _createObjectiveHelper_2() {
      final TransformationLengthDimension _transformationLengthDimension = new TransformationLengthDimension();
      return _transformationLengthDimension;
   }

   protected static double _createObjectiveHelper_rl_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.avgCombinedNorm((Plan) root);
   }

   protected static EGraph adaptInputGraph(final ModuleManager moduleManager, final EGraph initialGraph) {
      final EGraph problemGraph = MomotUtil.copy(initialGraph);
      // final EObject root = MomotUtil.getRoot(problemGraph);
      // return MomotUtil.createEGraph(adaptInputModel(root));
      return problemGraph;
   }

   protected static EObject adaptInputModel(final EObject root) {
      final Plan p = (Plan) root;
      // final int numSprintsCreated = p.getMinSprints();
      for(int i = 0; i < p.getBacklog().getWorkitems().size(); i++) {
         final Sprint newSprint = SprintPlanningFactory.eINSTANCE.createSprint();
         newSprint.setName("S_" + String.valueOf(i + 1));
         p.getSprints().add(newSprint);
      }
      final List<WorkItem> workItemList = p.getBacklog().getWorkitems();
      for(int i = 0; i < workItemList.size(); i++) {
         p.getSprints().get(i).getCommittedItem().add(workItemList.get(i));

      }

      p.setCurrentSprints(workItemList.size());
      return p;
   }

   // protected static EObject adaptInputModel(final EObject root) {
   // final Plan p = (Plan) root;
   // final int numSprintsCreated = p.getMinSprints();
   // for(int i = 0; i < p.getMinSprints(); i++) {
   // final Sprint newSprint = SprintPlanningFactory.eINSTANCE.createSprint();
   // newSprint.setName("S_" + String.valueOf(i + 1));
   // p.getSprints().add(newSprint);
   // }
   // final List<WorkItem> workItemList = p.getBacklog().getWorkitems();
   // for(int i = 0; i < workItemList.size(); i++) {
   // p.getSprints().get(i % numSprintsCreated).getCommittedItem().add(workItemList.get(i));
   //
   // }
   // // for(final Sprint sp : p.getSprints()) {
   // // for(final WorkItem wi : sp.getCommittedItem()) {
   // // System.out.print(wi.getName() + " @ " + sp.getName());
   // // System.out.print(", " + wi.getName() + " @ " + wi.getIsPlannedFor().get(0).getName() + " / ");
   // // }
   // // }
   // p.setCurrentSprints(numSprintsCreated);
   // return p;
   // }

   public static EGraph createInputGraph(final String initialGraph, final ModuleManager moduleManager) {
      final EGraph graph = moduleManager.loadGraph(initialGraph);
      return adaptInputGraph(moduleManager, graph);
   }

   public static void finalization() {
      System.out.println("Search finished.");

   }

   public static void initialization() {
      SprintPlanningPackage.eINSTANCE.eClass();
      System.out.println("Search started.");
   }

   public static void main(final String... args) {
      initialization();
      overridefromCMDArguments(args);
      //
      final SprintPlanningSearch search = new SprintPlanningSearch();
      search.performSearch(INITIAL_MODEL, SOLUTION_LENGTH);
      finalization();
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
         INITIAL_MODEL = "problem/" + CASE + ".xmi";
         NR_RUNS = Integer.parseInt(cmd.getOptionValue("runs"));
         MAX_EVALUATIONS = Integer.parseInt(cmd.getOptionValue("evaluations"));
      }

   }

   protected final String[] unitsToRemove = new String[] { "sprint-mutation::moveWorkItemToNewSprint",
         "sprint-mutation::deleteEmptySprint", "sprint-mutation::addUnassignedWorkItemToNewSprint",
         // "sprint-mutation::addUnassignedWorkItemToNewSprintRespectingMax",
         // "sprint-mutation::addUnassignedWorkItemToExSprint"
   };

   protected final String[] modules = new String[] { "transformations/sprint-mutation2-prioritized.henshin" };

   protected final int populationSize = 100;

   protected String baseName;

   protected double significanceLevel = 0.01;

   protected IFitnessDimension<TransformationSolution> _createConstraint_0(
         final TransformationSearchOrchestration orchestration) {
      return new AbstractEGraphFitnessDimension("noUnassignedWorkitems",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createConstraintHelper_0(solution, graph, root);
         }
      };
   }

   protected IFitnessDimension<TransformationSolution> _createConstraint_1(
         final TransformationSearchOrchestration orchestration) {
      return new AbstractEGraphFitnessDimension("noExceedingMaxNumOfSprints",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createConstraintHelper_1(solution, graph, root);
         }
      };
   }

   protected IFitnessDimension<TransformationSolution> _createConstraint_2(
         final TransformationSearchOrchestration orchestration) {
      return new AbstractEGraphFitnessDimension("noFallingBelowMinNumOfSprints",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createConstraintHelper_2(solution, graph, root);
         }
      };
   }

   protected IFitnessDimension<TransformationSolution> _createConstraint_3(
         final TransformationSearchOrchestration orchestration) {
      return new AbstractEGraphFitnessDimension("noMultiAssignedWorkItems",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createConstraintHelper_3(solution, graph, root);
         }
      };
   }

   protected double _createConstraintHelper_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.noUnassignedWorkitems((Plan) root);
   }

   protected double _createConstraintHelper_1(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.noExceedingMaxNumOfSprints((Plan) root);
   }

   protected double _createConstraintHelper_2(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.noFallingBelowMinNumOfSprints((Plan) root);
   }

   protected double _createConstraintHelper_3(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.noMultiAssignedWorkItems((Plan) root);
   }

   protected ProgressListener _createListener_0() {
      final SeedRuntimePrintListener _seedRuntimePrintListener = new SeedRuntimePrintListener();
      return _seedRuntimePrintListener;
   }

   protected IRegisteredAlgorithm<NSGAII> _createRegisteredAlgorithm_0(
         final TransformationSearchOrchestration orchestration,
         final EvolutionaryAlgorithmFactory<TransformationSolution> moea,
         final LocalSearchAlgorithmFactory<TransformationSolution> local) {
      final IRegisteredAlgorithm<NSGAII> _createNSGAII = moea.createNSGAII();
      return _createNSGAII;
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
      // function.addObjective(_createObjective_rl_0());

      function.addConstraint(_createConstraint_rl_0());
      function.addObjective(_createObjective_0());

      function.addObjective(_createObjective_1());
      function.addObjective(_createObjective_2());

      // function.addConstraint(_createConstraint_0(orchestration));
      // //
      // function.addConstraint(_createConstraint_1(orchestration));
      // function.addConstraint(_createConstraint_2(orchestration));
      // function.addConstraint(_createConstraint_3(orchestration));
      return function;
   }

   protected ModuleManager createModuleManager() {
      final ModuleManager manager = new ModuleManager();
      for(final String module : modules) {
         manager.addModule(URI.createFileURI(new File(module).getPath().toString()).toString());
      }
      manager.removeUnits(unitsToRemove);

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
            .createEvolutionaryAlgorithmFactory(populationSize);
      final LocalSearchAlgorithmFactory<TransformationSolution> local = orchestration
            .createLocalSearchAlgorithmFactory();

      // final IEGraphMultiDimensionalFitnessFunction ff = new EGraphMultiDimensionalFitnessFunction();

      final EnvironmentBuilder<TransformationSolution> envBuilder = new EnvironmentBuilder<>();

      final Map<IEnvironment.Type, IEnvironment<TransformationSolution>> env = envBuilder
            .fitnessAggregator((final double[] objectiveVals) -> {
               return objectiveVals[0] + objectiveVals[1];
               // }).build();
            })
            // .constraintAggregator((final double[] constraintVals) -> {
            // return constraintVals[0];
            // })

            // .fitnessScaler((final EObject m) -> {
            // final Plan p = (Plan) m;
            // final int noWorkitems = p.getBacklog().getWorkitems().size();
            // final double noAssignedWorkitems = p.getBacklog().getWorkitems().stream()
            // .filter(item -> !item.getIsPlannedFor().isEmpty()).count();
            // final double frac = noAssignedWorkitems / noWorkitems;
            // return Math.pow(frac, 2);
            // }).constraintScaler((final EObject m) -> {
            // final Plan p = (Plan) m;
            // final int noWorkitems = p.getBacklog().getWorkitems().size();
            // final double noAssignedWorkitems = p.getBacklog().getWorkitems().stream()
            // .filter(item -> !item.getIsPlannedFor().isEmpty()).count();
            // final double frac = noAssignedWorkitems / noWorkitems;
            // return 1. - frac;
            // }).build();
            .build();

      final RLAlgorithmFactory<TransformationSolution> rl = orchestration.createRLAlgorithmFactory(env);
      //
      // orchestration.addAlgorithm("QLearningExplore5", rl.createSingleObjectiveExploreQLearner(5, 1., 0.99, .95, true,
      // (.95 - .05) / MAX_EVALUATIONS, 0.05, "QExplore-" + CASE, 10, 0, null, null, false));

      orchestration.addAlgorithm("QLearningExplore", rl.createSingleObjectiveExploreQLearner(10, 1., 0.99, .95, true,
            (.95 - .05) / MAX_EVALUATIONS, 0.05, "QExplore-" + CASE, 10, 0, null, null, false));

      // orchestration.addAlgorithm("QLearningExplore1", rl.createSingleObjectiveExploreQLearner(1, 1., 0.99, .95, true,
      // (.95 - .05) / MAX_EVALUATIONS, 0.05, "QExplore-" + CASE, 10, 0, null, null, false));

      orchestration.addAlgorithm("NSGAII-2",
            moea.createNSGAII(new TournamentSelection(2), new OnePointCrossover(1.0),
                  new TransformationPlaceholderMutation(0.1),
                  new TransformationVariableMutation(orchestration.getSearchHelper(), 0.05)));

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
      final ISolutionWriter<TransformationSolution> solutionWriter = experiment.getSearchOrchestration()
            .createSolutionWriter();
      final IPopulationWriter<TransformationSolution> populationWriter = experiment.getSearchOrchestration()
            .createPopulationWriter();
      final TransformationResultManager resultManager = new TransformationResultManager(experiment);

      resultManager.saveParetosetPerRun(String.valueOf(MAX_EVALUATIONS), "output/populations/" + CASE);
      // experiment.getSearchOrchestration().getFitnessFunction().evaluate(resultManager.get)
      Population population;

      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save objectives of all algorithms to 'output/objectives/objective_values.txt'");
      SearchResultManager.saveObjectives("output/objectives/objective_values.txt", population);
      System.out.println("---------------------------");
      System.out.println("Objectives of all algorithms");
      System.out.println("---------------------------");
      System.out.println(SearchResultManager.printObjectives(population));

      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save solutions of all algorithms to 'output/solutions/objective_values.txt'");
      SearchResultManager.savePopulation("output/solutions/objective_values.txt", population, populationWriter);
      System.out.println("- Save solutions of all algorithms to 'output/solutions/objective_values.txt'");
      SearchResultManager.saveSolutions("output/solutions/", baseName,
            MomotUtil.asIterables(population, TransformationSolution.class), solutionWriter);

      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save models of all algorithms to 'output/models/'");
      TransformationResultManager.saveModels("output/models/", baseName, population);

      for(final Entry<SearchExecutor, List<NondominatedPopulation>> entry : resultManager.getResults().entrySet()) {

         System.out.println(entry.getKey().getName());

         population = SearchResultManager.createApproximationSet(experiment, entry.getKey().getName());
         System.out.println(SearchResultManager.printObjectives(population) + "\n");

         population = SearchResultManager.createApproximationSet(experiment, entry.getKey().getName());
         SearchResultManager.saveObjectives("output/objectives/" + entry.getKey().getName() + ".txt", population);
      }

      // for(final Entry<SearchExecutor, List<NondominatedPopulation>> entry : experiment.getResults().entrySet()) {
      // final String algName = entry.getKey().getName();
      // if(algName.compareTo("QLearningExplore") == 0) {
      // for(final NondominatedPopulation ndp : entry.getValue()) {
      // for(int i = 0; i < ndp.size(); i++) {
      // final double a = experiment.getSearchOrchestration().getFitnessFunction()
      // .evaluate((TransformationSolution) ndp.get(i));
      // }
      // }
      // }
      // final List<NondominatedPopulation> ndps = entry.getValue();
      // }

      return resultManager;
   }

   protected SearchAnalyzer performAnalysis(final SearchExperiment<TransformationSolution> experiment) {
      final SearchAnalysis analysis = new SearchAnalysis(experiment);
      analysis.setHypervolume(true);
      analysis.setGenerationalDistance(true);
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
      System.out.println("InputModel:      " + INITIAL_MODEL);
      System.out.println("Objectives:      " + orchestration.getFitnessFunction().getObjectiveNames());
      System.out.println("NrObjectives:    " + orchestration.getNumberOfObjectives());
      System.out.println("Constraints:     " + orchestration.getFitnessFunction().getConstraintNames());
      System.out.println("NrConstraints:   " + orchestration.getNumberOfConstraints());
      System.out.println("Transformations: " + Arrays.toString(modules));
      System.out.println("Units:           " + orchestration.getModuleManager().getUnits());
      System.out.println("SolutionLength:  " + orchestration.getSolutionLength());
      System.out.println("PopulationSize:  " + populationSize);
      System.out.println("Iterations:      " + MAX_EVALUATIONS / populationSize);
      System.out.println("MaxEvaluations:  " + MAX_EVALUATIONS);
      System.out.println("AlgorithmRuns:   " + NR_RUNS);
      System.out.println("---------------------------");
   }
}
