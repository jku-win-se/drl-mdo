package icmt.tool.momot.demo;

import at.ac.tuwien.big.moea.SearchAnalysis;
import at.ac.tuwien.big.moea.SearchExperiment;
import at.ac.tuwien.big.moea.SearchResultManager;
import at.ac.tuwien.big.moea.experiment.analyzer.SearchAnalyzer;
import at.ac.tuwien.big.moea.experiment.executor.SearchExecutor;
import at.ac.tuwien.big.moea.experiment.executor.listener.CurrentNondominatedPopulationPrintListener;
import at.ac.tuwien.big.moea.experiment.executor.listener.SeedRuntimePrintListener;
import at.ac.tuwien.big.moea.search.algorithm.EvolutionaryAlgorithmFactory;
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

import com.google.common.base.Objects;

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
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
import org.eclipse.xtext.xbase.lib.CollectionExtensions;
import org.eclipse.xtext.xbase.lib.ExclusiveRange;
import org.eclipse.xtext.xbase.lib.Functions.Function1;
import org.eclipse.xtext.xbase.lib.IterableExtensions;
import org.moeaframework.core.Algorithm;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.operator.OnePointCrossover;
import org.moeaframework.core.operator.TournamentSelection;
import org.moeaframework.util.progress.ProgressListener;

import icmt.tool.momot.demo.architecture.ArchitectureFactory;
import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.ClassModel;
import icmt.tool.momot.demo.architecture.Feature;

public class SOArchitectureSearch {

   protected static String CASE = "CRA_D";

   protected static int SOLUTION_LENGTH = 160;
   private static String INPUT_MODEL = "problem/" + CASE + ".xmi";

   protected static boolean PRINT_POPULATIONS = false;
   protected static String PRINT_DIRECTORY = "output/populations/" + CASE;

   protected static int MAX_EVALUATIONS = 100000;

   protected static int NR_RUNS = 1;

   protected final static String[] modules = new String[] { "transformations/architecture.henshin" };

   protected final static int POPULATION_SIZE = 100;

   protected static IFitnessDimension<TransformationSolution> _createObjective_0(
         final TransformationSearchOrchestration orchestration) {
      return new AbstractEGraphFitnessDimension("CRA-Index",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Maximum) {
         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createObjectiveHelper_0(solution, graph, root);
         }
      };
   }

   protected static IFitnessDimension<TransformationSolution> _createObjective_1(
         final TransformationSearchOrchestration orchestration) {
      return new AbstractEGraphFitnessDimension("Coupling",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {
         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createObjectiveHelper_1(solution, graph, root);
         }
      };
   }

   protected static IFitnessDimension<TransformationSolution> _createObjective_2(
         final TransformationSearchOrchestration orchestration) {
      final IFitnessDimension<TransformationSolution> dimension = _createObjectiveHelper_2();
      dimension.setName("SolutionLength");
      dimension.setFunctionType(at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum);
      return dimension;
   }

   protected static double _createObjectiveHelper_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessCalculator.calculateCRAIndex((ClassModel) root);
   }

   protected static double _createObjectiveHelper_1(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessCalculator.calculateCouplingNorm((ClassModel) root);
   }

   protected static IFitnessDimension<TransformationSolution> _createObjectiveHelper_2() {
      final TransformationLengthDimension _transformationLengthDimension = new TransformationLengthDimension();
      return _transformationLengthDimension;
   }

   protected static EGraph adaptInputGraph(final ModuleManager moduleManager, final EGraph initialGraph) {
      final EGraph problemGraph = MomotUtil.copy(initialGraph);
      final EObject root = MomotUtil.getRoot(problemGraph);
      return MomotUtil.createEGraph(adaptInputModel(root));
   }

   protected static EObject adaptInputModel(final EObject root) {
      final ClassModel cm = (ClassModel) root;
      final int _size = cm.getFeatures().size();
      final int _size_1 = cm.getClasses().size();
      final int _minus = _size - _size_1;
      final ExclusiveRange _doubleDotLessThan = new ExclusiveRange(0, _minus, true);
      for(final Integer i : _doubleDotLessThan) {
         {
            final icmt.tool.momot.demo.architecture.Class newClass = ArchitectureFactory.eINSTANCE.createClass();
            newClass.setName("Class_" + i);
            cm.getClasses().add(newClass);
         }
      }
      final EList<Feature> _features = cm.getFeatures();
      int i = 0;
      for(final Feature feature : _features) {
         final icmt.tool.momot.demo.architecture.Class _isEncapsulatedBy = feature.getIsEncapsulatedBy();
         final boolean _equals = Objects.equal(_isEncapsulatedBy, null);
         if(_equals) {
            cm.getClasses().get(i++).getEncapsulates().add(feature);
         }
      }
      return cm;
   }

   protected static IEGraphMultiDimensionalFitnessFunction createFitnessFunction(
         final TransformationSearchOrchestration orchestration) {
      final IEGraphMultiDimensionalFitnessFunction function = new EGraphMultiDimensionalFitnessFunction();
      function.addObjective(_createObjective_0(orchestration));
      // function.addObjective(_createObjective_1(orchestration));
      function.addObjective(_createObjective_2(orchestration));

      return function;
   }

   protected static EGraph createInputGraph(final String initialGraph, final ModuleManager moduleManager) {
      final EGraph graph = moduleManager.loadGraph(initialGraph);
      return adaptInputGraph(moduleManager, graph);
   }

   protected static ModuleManager createModuleManager() {
      final ModuleManager manager = new ModuleManager();
      for(final String module : modules) {
         manager.addModule(URI.createFileURI(new File(module).getPath().toString()).toString());
      }
      return manager;
   }

   public static TransformationSearchOrchestration createOrchestration(final String initialGraph,
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
      // final Map<IEnvironment.Type, IEnvironment<TransformationSolution>> env = envBuilder.singleObjective("CRAIndex")
      // .build();
      final EnvironmentBuilder<TransformationSolution> envBuilder = new EnvironmentBuilder<>();

      final Map<IEnvironment.Type, IEnvironment<TransformationSolution>> env = envBuilder
            .fitnessAggregator((final double[] objectiveVals) -> {
               return objectiveVals[0];
               // }).constraintAggregator((final double[] constraintVals) -> {
               // return constraintVals[0];
               // }).build();
            }).build();

      final RLAlgorithmFactory<TransformationSolution> rl = orchestration.createRLAlgorithmFactory(env);
      orchestration.addAlgorithm("QLearningExplore", rl.createSingleObjectiveExploreQLearner(10, 1., 0.99, 0.95, true,
            (0.95 - .05) / MAX_EVALUATIONS, 0.05, "QExplore-" + CASE, 10, 0, null, null, false));

      orchestration.addAlgorithm("NSGAII-2",
            moea.createNSGAII(new TournamentSelection(2), new OnePointCrossover(1.0),
                  new TransformationPlaceholderMutation(0.1),
                  new TransformationVariableMutation(orchestration.getSearchHelper(), 0.05)));

      // orchestration.addAlgorithm("QLearning", rl.createSingleObjectiveQLearner(0.9, 1.0, true,
      // (1.0 - .05) / MAX_EVALUATIONS, 0.05, "Q-" + CASE, 10, 0, null, null, false));

      return orchestration;
   }

   public static void initialization() {
      ArchitecturePackage.eINSTANCE.eClass();
      System.out.println("Search started.");
   }

   public static void main(final String... args) {
      initialization();

      overridefromCMDArguments(args);

      final SOArchitectureSearch search = new SOArchitectureSearch();
      search.performSearch(INPUT_MODEL, SOLUTION_LENGTH);
      System.out.println("Search finished");
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
         // System.out.println("Running without cmd arguments...");
      } else {
         CASE = String.valueOf(cmd.getOptionValue("model"));
         SOLUTION_LENGTH = Integer.parseInt(cmd.getOptionValue("solution_length"));
         INPUT_MODEL = "problem/" + CASE + ".xmi";
         PRINT_DIRECTORY = Paths.get(".", "output", "populations", CASE).toString();
         NR_RUNS = Integer.parseInt(cmd.getOptionValue("runs"));
         MAX_EVALUATIONS = Integer.parseInt(cmd.getOptionValue("evaluations"));
      }

   }

   protected String baseName;

   protected double significanceLevel = 0.01;

   protected ProgressListener _createListener_0() {
      final SeedRuntimePrintListener _seedRuntimePrintListener = new SeedRuntimePrintListener();
      return _seedRuntimePrintListener;
   }

   protected void adaptResultModel(final EObject root) {
      final ClassModel cm = (ClassModel) root;
      final Function1<icmt.tool.momot.demo.architecture.Class, Boolean> _function = (
            final icmt.tool.momot.demo.architecture.Class c) -> {
         final int _size = c.getEncapsulates().size();
         return Boolean.valueOf(_size == 0);
      };
      final Iterable<icmt.tool.momot.demo.architecture.Class> emptyClasses = IterableExtensions.<icmt.tool.momot.demo.architecture.Class> filter(
            cm.getClasses(), _function);
      CollectionExtensions.<icmt.tool.momot.demo.architecture.Class> removeAll(cm.getClasses(), emptyClasses);
   }

   public void adaptResultModels(final List<File> modelFiles) {
      final HenshinResourceSet set = new HenshinResourceSet();
      for(final File file : modelFiles) {
         final EGraph graph = MomotUtil.loadGraph(file.getPath());
         final EObject root = MomotUtil.getRoot(graph);
         adaptResultModel(root);
         MomotUtil.saveGraph(graph, file.getPath());
      }
   }

   protected SearchExperiment<TransformationSolution> createExperiment(
         final TransformationSearchOrchestration orchestration) {
      final SearchExperiment<TransformationSolution> experiment = new SearchExperiment<>(orchestration,
            MAX_EVALUATIONS);
      experiment.setNumberOfRuns(NR_RUNS);
      experiment.addProgressListener(_createListener_0());

      if(PRINT_POPULATIONS) {
         final List<String> algorithmNames = new ArrayList<>();
         for(final IRegisteredAlgorithm<? extends Algorithm> a : orchestration.getAlgorithms()) {
            algorithmNames.add(orchestration.getAlgorithmName(a));
         }

         if(!Files.exists(Paths.get(PRINT_DIRECTORY))) {
            new File(PRINT_DIRECTORY).mkdirs();
         }

         experiment.addProgressListener(new CurrentNondominatedPopulationPrintListener(PRINT_DIRECTORY, algorithmNames,
               NR_RUNS, 1000, MAX_EVALUATIONS));

      }

      return experiment;
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
      adaptResultModels(TransformationResultManager.saveModels("output/models/", baseName, population));

      System.out.println("- Save objectives of algorithms seperately to 'output/objectives/<algorithm>.txt'");
      System.out.println("- Save models of algorithms seperately to 'output/solutions/<algorithm>.txt'´\n");

      for(final Entry<SearchExecutor, List<NondominatedPopulation>> entry : resultManager.getResults().entrySet()) {

         System.out.println(entry.getKey().getName());

         population = SearchResultManager.createApproximationSet(experiment, entry.getKey().getName());
         System.out.println(SearchResultManager.printObjectives(population) + "\n");
         adaptResultModels(TransformationResultManager.saveModels("output/models/" + entry.getKey().getName(),
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
      // experiment.addCustomCollector(new ApproximationSetCollector());

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
      // System.out.println("InputModel: " + INITIAL_MODEL);
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
}
