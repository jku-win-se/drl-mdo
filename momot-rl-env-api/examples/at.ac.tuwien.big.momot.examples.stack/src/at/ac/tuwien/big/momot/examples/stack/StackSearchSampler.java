package at.ac.tuwien.big.momot.examples.stack;

import at.ac.tuwien.big.moea.SearchAnalysis;
import at.ac.tuwien.big.moea.SearchExperiment;
import at.ac.tuwien.big.moea.SearchResultManager;
import at.ac.tuwien.big.moea.experiment.analyzer.SearchAnalyzer;
import at.ac.tuwien.big.moea.experiment.analyzer.kneepoint.PopulationAnalyzer;
import at.ac.tuwien.big.moea.experiment.executor.listener.SeedRuntimePrintListener;
import at.ac.tuwien.big.moea.print.IPopulationWriter;
import at.ac.tuwien.big.moea.print.ISolutionWriter;
import at.ac.tuwien.big.moea.search.algorithm.EvolutionaryAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.LocalSearchAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.provider.IRegisteredAlgorithm;
import at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension;
import at.ac.tuwien.big.moea.util.MathUtil;
import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.TransformationResultManager;
import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.TransformationStateExplorer;
import at.ac.tuwien.big.momot.examples.stack.stack.Stack;
import at.ac.tuwien.big.momot.examples.stack.stack.StackModel;
import at.ac.tuwien.big.momot.examples.stack.stack.StackPackage;
import at.ac.tuwien.big.momot.learning.dsutils.TransformationSample;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.problem.solution.variable.ITransformationVariable;
import at.ac.tuwien.big.momot.problem.unit.parameter.IParameterValue;
import at.ac.tuwien.big.momot.problem.unit.parameter.random.RandomIntegerValue;
import at.ac.tuwien.big.momot.search.algorithm.operator.mutation.TransformationParameterMutation;
import at.ac.tuwien.big.momot.search.algorithm.operator.mutation.TransformationPlaceholderMutation;
import at.ac.tuwien.big.momot.search.algorithm.operator.mutation.TransformationVariableMutation;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.ApplicationState;
import at.ac.tuwien.big.momot.search.criterion.MinimumObjectiveCondition;
import at.ac.tuwien.big.momot.search.fitness.EGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.dimension.AbstractEGraphFitnessDimension;
import at.ac.tuwien.big.momot.search.fitness.dimension.TransformationLengthDimension;
import at.ac.tuwien.big.momot.search.solution.repair.ITransformationRepairer;
import at.ac.tuwien.big.momot.search.solution.repair.TransformationPlaceholderRepairer;
import at.ac.tuwien.big.momot.util.MomotUtil;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.interpreter.UnitApplication;
import org.eclipse.emf.henshin.model.Parameter;
import org.eclipse.emf.henshin.model.Rule;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
import org.eclipse.xtext.xbase.lib.Functions.Function1;
import org.eclipse.xtext.xbase.lib.ListExtensions;
import org.moeaframework.algorithm.NSGAII;
import org.moeaframework.algorithm.RandomSearch;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.operator.OnePointCrossover;
import org.moeaframework.core.operator.TournamentSelection;
import org.moeaframework.util.progress.ProgressListener;

@SuppressWarnings("all")
public class StackSearchSampler {
   protected static final String outName = "5";
   protected static final String MODELSET_PATH = "modelset/input/5_mutations";
   protected static final int SOLUTION_LENGTH = 7;
   protected static final double STD_SEARCH_THRESHOLD = .5;
   private static final Integer SHIFT_MAX = 5;

   protected final static String[] modules = new String[] { "model/stack.henshin" };
   protected final static String _parameterValueKey_0 = StackModule.ShiftLeft.Parameter.AMOUNT;

   protected final static String _parameterValueKey_1 = StackModule.ShiftRight.Parameter.AMOUNT;

   protected final static String[] nonSolutionParameters = new String[] { StackModule.ShiftLeft.Parameter.FROM_LOAD,
         StackModule.ShiftLeft.Parameter.TO_LOAD, StackModule.ShiftRight.Parameter.FROM_LOAD,
         StackModule.ShiftRight.Parameter.TO_LOAD };

   protected static IParameterValue<?> _createParameterValue_0() {
      final RandomIntegerValue _randomIntegerValue = new RandomIntegerValue(1, SHIFT_MAX);
      return _randomIntegerValue;
   }

   protected static IParameterValue<?> _createParameterValue_1() {
      final RandomIntegerValue _randomIntegerValue = new RandomIntegerValue(1, SHIFT_MAX);
      return _randomIntegerValue;
   }

   protected static ModuleManager createModuleManager() {
      final ModuleManager manager = new ModuleManager();
      for(final String module : modules) {
         manager.addModule(URI.createFileURI(new File(module).getPath().toString()).toString());
      }
      manager.addNonSolutionParameters(nonSolutionParameters);
      manager.setParameterValue(_parameterValueKey_0, _createParameterValue_0());
      manager.setParameterValue(_parameterValueKey_1, _createParameterValue_1());
      return manager;
   }

   private static void deleteFolder(final File folder) {
      final File[] files = folder.listFiles();
      if(files != null) { // some JVMs return null for empty dirs
         for(final File f : files) {
            if(f.isDirectory()) {
               deleteFolder(f);
            } else {
               f.delete();
            }
         }
      }
      folder.delete();
   }

   public static void finalization() {
      System.out.println("Search finished.");
   }

   public static void initialization() {
      StackPackage.eINSTANCE.getClass();
   }

   public static void main(final String... args) {
      initialization();
      final StackSearchSampler search = new StackSearchSampler();
      final ObjectMapper objectMapper = new ObjectMapper();
      final List<TransformationSample> samples = new ArrayList<>();
      final ModuleManager m = createModuleManager();
      final Set<Parameter> nsParams = m.getNonSolutionParameters();
      final List<String> unitNames = new ArrayList<>();
      final Map<Integer, List<String>> ruleIdToParamNames = new HashMap<>();
      for(final org.eclipse.emf.henshin.model.Module mod : m.getModules()) {

         for(final Rule r : mod.getAllRules()) {
            unitNames.add(r.getName());
            final List<String> paramNameList = new ArrayList<>();
            for(final Parameter p : r.getParameters()) {
               if(!nsParams.contains(p)) {
                  paramNameList.add(p.getName());
               }
            }
            ruleIdToParamNames.put(unitNames.indexOf(r.getName()), paramNameList);
         }
      }
      deleteFolder(new File("output/ds/"));
      final File[] files = new File(MODELSET_PATH).listFiles();
      Arrays.sort(files);
      final List<File> fileList = Arrays.stream(files).filter(f -> !f.isDirectory()).collect(Collectors.toList());
      int fileNr = 0;
      for(final File f : fileList) {
         System.out.println(f.getAbsolutePath());
         final HenshinResourceSet henshinResourceSet = new HenshinResourceSet();
         final Resource modelResource = henshinResourceSet.getResource(f.toString());
         final EGraph graph = MomotUtil.eGraphOf(modelResource, true);
         final Population p = search.performSearch(f.getAbsolutePath(), SOLUTION_LENGTH);

         int idx = 0;
         double curStd = p.get(0).getObjective(0);
         for(int i = 1; i < p.size(); i++) {
            if(p.get(i).getObjective(0) < curStd) {
               curStd = p.get(i).getObjective(0);
               idx = i;

            }
         }

         int transformationNr = 0;
         final List<ITransformationVariable> sVars = ((TransformationSolution) p.get(idx)).getVariablesAsList(true);
         for(final ITransformationVariable tv : sVars) {

            final TransformationSolution curSolution = new TransformationSolution(MomotUtil.copy(graph),
                  sVars.subList(0, transformationNr), 1);
            curSolution.execute();
            final String modelPath = TransformationResultManager.saveModel("output/ds/" + outName,
                  String.format("model-%d_%d", fileNr, transformationNr), curSolution);

            printSM(MomotUtil.getRoot(curSolution.getResultGraph(), StackModel.class));
            System.out.println("------------------------------------------");

            final boolean isLast = transformationNr++ == sVars.size() - 1;
            final UnitApplication ua = tv;
            final String uaName = ua.getUnit().getName();

            final ApplicationState as = ApplicationState.of(tv, ruleIdToParamNames.get(unitNames.indexOf(uaName)),
                  unitNames.indexOf(uaName));
            samples.add(TransformationSample.of(
                  modelPath.replace("output/ds/" + outName, "datasets/stack/train/" + outName), as, isLast, 0, null));
         }

         final TransformationSolution finalSolution = new TransformationSolution(MomotUtil.copy(graph), sVars, 1);
         finalSolution.execute();
         printSM(MomotUtil.getRoot(finalSolution.getResultGraph(), StackModel.class));

         fileNr++;

      }

      try {
         objectMapper.writeValue(new File("output/ds/" + outName + "/ds.json"), samples);
      } catch(final IOException e) {
         e.printStackTrace();
      }
      finalization();
   }

   public static void printSM(final StackModel sm) {
      final Map<String, String> nodeStates = new HashMap<>();
      for(final Stack s : sm.getStacks()) {
         System.out.format(" %d ", s.getLoad());
      }
      System.out.println();
   }

   protected final ITransformationRepairer solutionRepairer = new TransformationPlaceholderRepairer();

   protected final int populationSize = 150;

   protected final int maxEvaluations = 500000;

   protected final int nrRuns = 1;

   protected String baseName;

   protected double significanceLevel = 0.01;

   protected ProgressListener _createListener_0() {
      final SeedRuntimePrintListener _seedRuntimePrintListener = new SeedRuntimePrintListener();
      return _seedRuntimePrintListener;
   }

   protected IFitnessDimension<TransformationSolution> _createObjective_0() {
      return new AbstractEGraphFitnessDimension("StandardDeviation",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {
         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createObjectiveHelper_0(solution, graph, root);
         }
      };
   }

   protected IFitnessDimension<TransformationSolution> _createObjective_1() {
      final IFitnessDimension<TransformationSolution> dimension = _createObjectiveHelper_1();
      dimension.setName("SolutionLength");
      dimension.setFunctionType(at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum);
      return dimension;
   }

   protected double _createObjectiveHelper_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {

      final Function1<Stack, Integer> _function = (final Stack it) -> {
         return Integer.valueOf(it.getLoad());
      };
      return MathUtil
            .getStandardDeviation(ListExtensions.<Stack, Integer> map(((StackModel) root).getStacks(), _function));
   }

   protected IFitnessDimension<TransformationSolution> _createObjectiveHelper_1() {
      final TransformationLengthDimension _transformationLengthDimension = new TransformationLengthDimension();
      return _transformationLengthDimension;
   }

   protected IRegisteredAlgorithm<RandomSearch> _createRegisteredAlgorithm_0(
         final TransformationSearchOrchestration orchestration,
         final EvolutionaryAlgorithmFactory<TransformationSolution> moea,
         final LocalSearchAlgorithmFactory<TransformationSolution> local) {
      final IRegisteredAlgorithm<RandomSearch> _createRandomSearch = moea.createRandomSearch();
      return _createRandomSearch;
   }

   protected IRegisteredAlgorithm<NSGAII> _createRegisteredAlgorithm_1(
         final TransformationSearchOrchestration orchestration,
         final EvolutionaryAlgorithmFactory<TransformationSolution> moea,
         final LocalSearchAlgorithmFactory<TransformationSolution> local) {
      final TournamentSelection _tournamentSelection = new TournamentSelection(2);
      final OnePointCrossover _onePointCrossover = new OnePointCrossover(1.0);
      final TransformationPlaceholderMutation _transformationPlaceholderMutation = new TransformationPlaceholderMutation(
            0.15);
      final ModuleManager _moduleManager = orchestration.getModuleManager();
      final TransformationParameterMutation _transformationParameterMutation = new TransformationParameterMutation(0.1,
            _moduleManager);
      final IRegisteredAlgorithm<NSGAII> _createNSGAII = moea.createNSGAII(_tournamentSelection, _onePointCrossover,
            _transformationPlaceholderMutation, _transformationParameterMutation);
      return _createNSGAII;
   }

   protected IRegisteredAlgorithm<NSGAII> _createRegisteredAlgorithm_2(
         final TransformationSearchOrchestration orchestration,
         final EvolutionaryAlgorithmFactory<TransformationSolution> moea,
         final LocalSearchAlgorithmFactory<TransformationSolution> local) {
      final TournamentSelection _tournamentSelection = new TournamentSelection(2);
      final OnePointCrossover _onePointCrossover = new OnePointCrossover(1.0);
      final TransformationPlaceholderMutation _transformationPlaceholderMutation = new TransformationPlaceholderMutation(
            0.15);
      final ModuleManager _moduleManager = orchestration.getModuleManager();
      final TransformationParameterMutation _transformationParameterMutation = new TransformationParameterMutation(0.1,
            _moduleManager);
      final IRegisteredAlgorithm<NSGAII> _createNSGAIII = moea.createNSGAIII(4, _tournamentSelection,
            _onePointCrossover, _transformationPlaceholderMutation, _transformationParameterMutation);
      return _createNSGAIII;
   }

   protected SearchExperiment<TransformationSolution> createExperiment(
         final TransformationSearchOrchestration orchestration) {
      final SearchExperiment<TransformationSolution> experiment = new SearchExperiment<>(orchestration, maxEvaluations,
            MinimumObjectiveCondition.create(Map.of(0, STD_SEARCH_THRESHOLD)));
      experiment.setNumberOfRuns(nrRuns);
      experiment.addProgressListener(_createListener_0());
      return experiment;
   }

   protected IEGraphMultiDimensionalFitnessFunction createFitnessFunction() {
      final IEGraphMultiDimensionalFitnessFunction function = new EGraphMultiDimensionalFitnessFunction();
      function.addObjective(_createObjective_0());
      function.addObjective(_createObjective_1());
      function.setSolutionRepairer(solutionRepairer);
      return function;
   }

   protected EGraph createInputGraph(final String initialGraph, final ModuleManager moduleManager) {
      final EGraph graph = moduleManager.loadGraph(initialGraph);
      return graph;
   }

   protected TransformationSearchOrchestration createOrchestration(final String initialGraph,
         final int solutionLength) {
      final TransformationSearchOrchestration orchestration = new TransformationSearchOrchestration();
      final ModuleManager moduleManager = createModuleManager();
      final EGraph graph = createInputGraph(initialGraph, moduleManager);
      orchestration.setModuleManager(moduleManager);
      orchestration.setProblemGraph(graph);
      orchestration.setSolutionLength(solutionLength);
      orchestration.setFitnessFunction(createFitnessFunction());

      final EvolutionaryAlgorithmFactory<TransformationSolution> moea = orchestration
            .createEvolutionaryAlgorithmFactory(populationSize);
      final LocalSearchAlgorithmFactory<TransformationSolution> local = orchestration
            .createLocalSearchAlgorithmFactory();
      // orchestration.addAlgorithm("Random", _createRegisteredAlgorithm_0(orchestration, moea, local));
      orchestration.addAlgorithm("NSGA-II", moea.createNSGAII(new TournamentSelection(2),
            // new RetiringSolutionVariation(orchestration.getSearchHelper(), 50, populationSize, 90),
            new OnePointCrossover(.8), new TransformationParameterMutation(0.2, orchestration.getModuleManager()),
            new TransformationVariableMutation(orchestration.getSearchHelper(), 0.2),
            new TransformationPlaceholderMutation(0.1)));
      // orchestration.addAlgorithm("NSGA_III", _createRegisteredAlgorithm_2(orchestration, moea, local));

      return orchestration;
   }

   protected TransformationStateExplorer createStateExploration(final String initialGraph,
         final ModuleManager moduleManager, final IEGraphMultiDimensionalFitnessFunction fitness) {
      final TransformationStateExplorer tse = new TransformationStateExplorer(moduleManager.getModules().get(0),
            initialGraph, fitness);
      return tse;
   }

   protected void deriveBaseName(final TransformationSearchOrchestration orchestration) {
      final EObject root = MomotUtil.getRoot(orchestration.getProblemGraph());
      if(root == null || root.eResource() == null || root.eResource().getURI() == null) {
         baseName = getClass().getSimpleName();
      } else {
         baseName = root.eResource().getURI().trimFileExtension().lastSegment();
      }
   }

   protected TransformationResultManager handleResults(final SearchExperiment<TransformationSolution> experiment,
         final TransformationStateExplorer exploration) {
      final ISolutionWriter<TransformationSolution> solutionWriter = experiment.getSearchOrchestration()
            .createSolutionWriter();
      final IPopulationWriter<TransformationSolution> populationWriter = experiment.getSearchOrchestration()
            .createPopulationWriter();
      final TransformationResultManager resultManager = new TransformationResultManager(experiment);
      Population population;
      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      // System.out.println("---------------------------");
      // System.out.println("Objectives of all algorithms");
      // System.out.println("---------------------------");
      // System.out.println(SearchResultManager.printObjectives(population));

      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      // System.out.println("- Save objectives of all algorithms to 'output/overall_objectives.pf'");
      SearchResultManager.saveObjectives("output/overall_objectives.pf", population);
      // System.out.println("---------------------------");
      // System.out.println("Objectives of all algorithms");
      // System.out.println("---------------------------");
      // System.out.println(SearchResultManager.printObjectives(population));

      population = SearchResultManager.createApproximationSet(experiment, "NSGA_III", "NSGA_II");
      // System.out.println("- Save objectives of 'NSGA_III', 'NSGA_II' to 'output/moea_objectives.pf'");
      SearchResultManager.saveObjectives("output/moea_objectives.pf", population);
      // System.out.println("---------------------------");
      // System.out.println("Objectives of 'NSGA_III', 'NSGA_II'");
      // System.out.println("---------------------------");
      // System.out.println(SearchResultManager.printObjectives(population));

      population = new PopulationAnalyzer(SearchResultManager.createApproximationSet(experiment)).getKneePointSolutions(
            SearchResultManager.createApproximationSet(experiment, "NSGA_III", "NSGA_II"), 3,
            PopulationAnalyzer.FitnessComparison.PROPER_UTILITY);
      // System.out.println(
      // "- Save Kneepoint Objectives of 'NSGA_III', 'NSGA_II' with 3 neighbors to
      // 'output/moea_kneepoints_objectives.pf'");
      // SearchResultManager.saveObjectives("output/moea_kneepoints_objectives.pf", population);
      // System.out.println("---------------------------");
      // System.out.println("Kneepoint Objectives of 'NSGA_III', 'NSGA_II' with 3 neighbors");
      // System.out.println("---------------------------");
      // System.out.println(SearchResultManager.printObjectives(population));

      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      // System.out.println("- Save solutions of all algorithms to 'output/overall_solutions.txt'");
      SearchResultManager.savePopulation("output/overall_solutions.txt", population, populationWriter);
      // System.out.println("- Save solutions of all algorithms to 'output/overall_solutions.txt'");
      SearchResultManager.saveSolutions("output/solutions/all/", baseName,
            MomotUtil.asIterables(population, TransformationSolution.class), solutionWriter);

      population = SearchResultManager.createApproximationSet(experiment, "NSGA_III", "NSGA_II");
      // System.out.println("- Save solutions of 'NSGA_III', 'NSGA_II' to 'output/moea_solutions.txt'");
      SearchResultManager.savePopulation("output/moea_solutions.txt", population, populationWriter);
      // System.out.println("- Save solutions of 'NSGA_III', 'NSGA_II' to 'output/moea_solutions.txt'");
      SearchResultManager.saveSolutions("output/solutions/moea/", baseName,
            MomotUtil.asIterables(population, TransformationSolution.class), solutionWriter);

      population = new PopulationAnalyzer(SearchResultManager.createApproximationSet(experiment)).getKneePointSolutions(
            SearchResultManager.createApproximationSet(experiment, "NSGA_III", "NSGA_II"), populationSize - 1,
            PopulationAnalyzer.FitnessComparison.PROPER_UTILITY);
      // System.out.println(
      // "- Save Kneepoint solutions of 'NSGA_III', 'NSGA_II' with maximum neighbors to
      // 'output/solutions/moea_kneepoint_solutions.txt'");
      SearchResultManager.savePopulation("output/solutions/moea_kneepoint_solutions.txt", population, populationWriter);
      // System.out
      // .println("- Save solutions of 'NSGA_III', 'NSGA_II' to 'output/solutions/moea_kneepoint_solutions.txt'");
      SearchResultManager.saveSolutions("output/solutions/moea/kneepoints/", baseName,
            MomotUtil.asIterables(population, TransformationSolution.class), solutionWriter);
      // System.out.println("---------------------------");
      // System.out.println("Kneepoint solutions of 'NSGA_III', 'NSGA_II' with maximum neighbors");
      // System.out.println("---------------------------");
      SearchResultManager.printPopulation(population, populationWriter);

      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      // System.out.println("- Save models of all algorithms to 'output/models/testing ladida'");
      TransformationResultManager.saveModels("output/models/testing ladida", baseName, population);
      // System.out.println("---------------------------");
      // System.out.println("Models of all algorithms");
      // System.out.println("---------------------------");
      // System.out.println(TransformationResultManager.printModels(population));

      population = SearchResultManager.createApproximationSet(experiment, "NSGA_III", "NSGA_II");
      // System.out.println("- Save models of 'NSGA_III', 'NSGA_II' to 'output/models/moea/'");
      TransformationResultManager.saveModels("output/models/moea/", baseName, population);

      population = SearchResultManager.createApproximationSet(experiment, "Random");
      // System.out.println("- Save models of 'Random' to 'output/models/random/'");
      TransformationResultManager.saveModels("output/models/random/", baseName, population);

      return resultManager;
   }

   protected SearchAnalyzer performAnalysis(final SearchExperiment<TransformationSolution> experiment) {
      final SearchAnalysis analysis = new SearchAnalysis(experiment);
      analysis.setAdditiveEpsilonIndicator(true);
      analysis.setHypervolume(true);
      analysis.setMaximumParetoFrontError(true);
      analysis.setShowAggregate(true);
      analysis.setShowIndividualValues(true);
      analysis.setShowStatisticalSignificance(true);
      analysis.setSignificanceLevel(significanceLevel);
      final SearchAnalyzer searchAnalyzer = analysis.analyze();
      return searchAnalyzer;
   }

   public Population performSearch(final String initialGraph, final int solutionLength) {
      final ModuleManager moduleManager = createModuleManager();
      final IEGraphMultiDimensionalFitnessFunction f = createFitnessFunction();
      final TransformationSearchOrchestration orchestration = createOrchestration(initialGraph, solutionLength);
      final TransformationStateExplorer exploration = null;
      // exploration = createStateExploration(initialGraph, moduleManager, f);
      deriveBaseName(orchestration);
      printSearchInfo(orchestration);
      final SearchExperiment<TransformationSolution> experiment = createExperiment(orchestration);
      experiment.run();
      // exploration.run();
      final NondominatedPopulation population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      // System.out.println("-------------------------------------------------------");
      // System.out.println("Analysis");
      // System.out.println("-------------------------------------------------------");
      // // performAnalysis(experiment);
      // System.out.println("-------------------------------------------------------");
      // System.out.println("Results");
      // System.out.println("-------------------------------------------------------");
      // // handleResults(experiment, exploration);
      return population;
   }

   // public void performSearch(final String initialGraph, final int solutionLength) {
   // final TransformationSearchOrchestration orchestration = createOrchestration(initialGraph, solutionLength);
   // deriveBaseName(orchestration);
   // printSearchInfo(orchestration);
   // final SearchExperiment<TransformationSolution> experiment = createExperiment(orchestration);
   // experiment.run();
   // System.out.println("-------------------------------------------------------");
   // System.out.println("Analysis");
   // System.out.println("-------------------------------------------------------");
   // performAnalysis(experiment);
   // System.out.println("-------------------------------------------------------");
   // System.out.println("Results");
   // System.out.println("-------------------------------------------------------");
   // handleResults(experiment);
   // }

   public void printSearchInfo(final TransformationSearchOrchestration orchestration) {
      System.out.println("-------------------------------------------------------");
      System.out.println("Search");
      System.out.println("-------------------------------------------------------");
      System.out.println("Objectives:      " + orchestration.getFitnessFunction().getObjectiveNames());
      System.out.println("NrObjectives:    " + orchestration.getNumberOfObjectives());
      System.out.println("Constraints:     " + orchestration.getFitnessFunction().getConstraintNames());
      System.out.println("NrConstraints:   " + orchestration.getNumberOfConstraints());
      System.out.println("Transformations: " + Arrays.toString(modules));
      System.out.println("Units:           " + orchestration.getModuleManager().getUnits());
      System.out.println("SolutionLength:  " + orchestration.getSolutionLength());
      System.out.println("PopulationSize:  " + populationSize);
      System.out.println("Iterations:      " + maxEvaluations / populationSize);
      System.out.println("MaxEvaluations:  " + maxEvaluations);
      System.out.println("AlgorithmRuns:   " + nrRuns);
      System.out.println("---------------------------");
   }
}
