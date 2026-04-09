package icmt.tool.momot.demo;

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
import at.ac.tuwien.big.moea.search.algorithm.provider.AbstractRegisteredAlgorithm;
import at.ac.tuwien.big.moea.search.algorithm.provider.IRegisteredAlgorithm;
import at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension;
import at.ac.tuwien.big.moea.util.MathUtil;
import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.TransformationResultManager;
import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.learning.dsutils.TransformationSample;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.problem.solution.variable.ITransformationVariable;
import at.ac.tuwien.big.momot.search.algorithm.operator.mutation.TransformationPlaceholderMutation;
import at.ac.tuwien.big.momot.search.algorithm.operator.mutation.TransformationVariableMutation;
import at.ac.tuwien.big.momot.search.fitness.EGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.dimension.AbstractEGraphFitnessDimension;
import at.ac.tuwien.big.momot.search.fitness.dimension.TransformationLengthDimension;
import at.ac.tuwien.big.momot.util.MomotUtil;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Objects;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.interpreter.UnitApplication;
import org.eclipse.emf.henshin.model.Parameter;
import org.eclipse.emf.henshin.model.Rule;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
import org.eclipse.xtext.xbase.lib.CollectionExtensions;
import org.eclipse.xtext.xbase.lib.ExclusiveRange;
import org.eclipse.xtext.xbase.lib.Functions.Function1;
import org.eclipse.xtext.xbase.lib.IterableExtensions;
import org.moeaframework.algorithm.EpsilonMOEA;
import org.moeaframework.algorithm.NSGAII;
import org.moeaframework.algorithm.RandomSearch;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.operator.OnePointCrossover;
import org.moeaframework.core.operator.TournamentSelection;
import org.moeaframework.util.progress.ProgressListener;

import icmt.tool.momot.demo.architecture.ArchitectureFactory;
import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.ClassModel;
import icmt.tool.momot.demo.architecture.Feature;

@SuppressWarnings("all")
public class ArchitectureSearcherSampler {
   protected static final String INITIAL_MODEL = "problem/nn/CRA_B_classes.xmi";

   protected static final int SOLUTION_LENGTH = 10;

   protected static final String outName = "cra_a";
   protected static final String MODELSET_PATH = "problem/nn";

   protected final static String[] modules = new String[] { "transformations/architecture.henshin" };

   protected final static int populationSize = 150;

   protected static IFitnessDimension<TransformationSolution> _createObjective_0(
         final TransformationSearchOrchestration orchestration) {
      return new AbstractEGraphFitnessDimension("CRAIndex",
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
      return new AbstractEGraphFitnessDimension("CohesionRatio",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Maximum) {
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
      return FitnessCalculator.calculateCohesion((ClassModel) root);
   }

   protected static IFitnessDimension<TransformationSolution> _createObjectiveHelper_2() {
      final TransformationLengthDimension _transformationLengthDimension = new TransformationLengthDimension();
      return _transformationLengthDimension;
   }

   protected static EGraph adaptInputGraph(final ModuleManager moduleManager, final EGraph initialGraph) {
      final EGraph problemGraph = MomotUtil.copy(initialGraph);
      final EObject root = MomotUtil.getRoot(problemGraph);
      return MomotUtil.createEGraph(root);
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
      for(final Feature feature : _features) {
         final icmt.tool.momot.demo.architecture.Class _isEncapsulatedBy = feature.getIsEncapsulatedBy();
         final boolean _equals = Objects.equal(_isEncapsulatedBy, null);
         if(_equals) {
            cm.getClasses().get(MathUtil.randomInteger(cm.getClasses().size())).getEncapsulates().add(feature);
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
      ArchitecturePackage.eINSTANCE.eClass();
      System.out.println("Search started.");
   }

   public static void main(final String... args) {
      initialization();
      final ArchitectureSearcherSampler search = new ArchitectureSearcherSampler();
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
      try {
         Files.createDirectories(Path.of("output/ds/", outName));
      } catch(final IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
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
         double craindex = p.get(0).getObjective(0);
         for(int i = 1; i < p.size(); i++) {
            if(p.get(i).getObjective(0) < craindex) {
               craindex = p.get(i).getObjective(0);
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

            printSM(MomotUtil.getRoot(curSolution.getResultGraph(), ClassModel.class));
            System.out.println("------------------------------------------");

            final boolean isLast = transformationNr++ == sVars.size() - 1;
            final UnitApplication ua = tv;
            final String uaName = ua.getUnit().getName();

            final at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.ApplicationState as = at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.ApplicationState
                  .of(tv, ruleIdToParamNames.get(unitNames.indexOf(uaName)), unitNames.indexOf(uaName));
            samples.add(TransformationSample
                  .of(modelPath.replace("output/ds/" + outName, "datasets/cra/train/" + outName), as, isLast, 0, null));
         }

         final TransformationSolution finalSolution = new TransformationSolution(MomotUtil.copy(graph), sVars, 1);
         finalSolution.execute();
         printSM(MomotUtil.getRoot(finalSolution.getResultGraph(), ClassModel.class));

         fileNr++;

      }

      try {
         objectMapper.writeValue(new File("output/ds/" + outName + "/ds.json"), samples);
      } catch(final IOException e) {
         e.printStackTrace();
      }
      finalization();
   }

   public static void printSM(final ClassModel sm) {
      final StringBuilder sb = new StringBuilder();
      for(final icmt.tool.momot.demo.architecture.Class c : sm.getClasses()) {
         final String classFeaturesCsv = c.getEncapsulates().stream().map(f -> f.getName())
               .collect(Collectors.joining(","));
         System.out.println("Class: " + classFeaturesCsv);

      }
   }

   protected final int maxEvaluations = 20000;

   protected final int nrRuns = 5;

   protected String baseName;

   protected double significanceLevel = 0.01;

   protected ProgressListener _createListener_0() {
      final SeedRuntimePrintListener _seedRuntimePrintListener = new SeedRuntimePrintListener();
      return _seedRuntimePrintListener;
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
      final IRegisteredAlgorithm<NSGAII> _createNSGAIII = moea.createNSGAIII();
      return _createNSGAIII;
   }

   protected AbstractRegisteredAlgorithm<EpsilonMOEA> _createRegisteredAlgorithm_2(
         final TransformationSearchOrchestration orchestration,
         final EvolutionaryAlgorithmFactory<TransformationSolution> moea,
         final LocalSearchAlgorithmFactory<TransformationSolution> local) {
      final AbstractRegisteredAlgorithm<EpsilonMOEA> _createEpsilonMOEA = moea.createEpsilonMOEA();
      return _createEpsilonMOEA;
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

   protected void adaptResultModels(final List<File> modelFiles) {
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
      final SearchExperiment<TransformationSolution> experiment = new SearchExperiment<>(orchestration, maxEvaluations);
      experiment.setNumberOfRuns(nrRuns);
      experiment.addProgressListener(_createListener_0());
      return experiment;
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
      //
      // orchestration.addAlgorithm("random", moea.createRandomSearch());
      //
      // orchestration.addAlgorithm("emoea", moea.createEpsilonMOEA());
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
      Population population;
      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save objectives of all algorithms to 'output/objectives/objective_values.txt'");
      SearchResultManager.saveObjectives("output/objectives/objective_values.txt", population);
      System.out.println("---------------------------");
      System.out.println("Objectives of all algorithms");
      System.out.println("---------------------------");
      System.out.println(SearchResultManager.printObjectives(population));

      population = SearchResultManager.createApproximationSet(experiment, "Random");
      System.out.println("- Save objectives of 'Random' to 'output/objectives/random_objective_values.txt'");
      SearchResultManager.saveObjectives("output/objectives/random_objective_values.txt", population);

      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save solutions of all algorithms to 'output/solutions/all_solutions.txt'");
      SearchResultManager.savePopulation("output/solutions/all_solutions.txt", population, populationWriter);
      System.out.println("- Save solutions of all algorithms to 'output/solutions/all_solutions.txt'");
      SearchResultManager.saveSolutions("output/solutions/", baseName,
            MomotUtil.asIterables(population, TransformationSolution.class), solutionWriter);

      population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save models of all algorithms to 'output/models/'");
      adaptResultModels(TransformationResultManager.saveModels("output/models/", baseName, population));

      population = new PopulationAnalyzer(SearchResultManager.createApproximationSet(experiment)).getKneePointSolutions(
            SearchResultManager.createApproximationSet(experiment, (String[]) null), populationSize - 1,
            PopulationAnalyzer.FitnessComparison.PROPER_UTILITY);
      System.out
            .println("- Save Kneepoint models of all algorithms with maximum neighbors to 'output/models/kneepoints/'");
      adaptResultModels(TransformationResultManager.saveModels("output/models/kneepoints/", baseName, population));

      return resultManager;
   }

   protected SearchAnalyzer performAnalysis(final SearchExperiment<TransformationSolution> experiment) {
      final SearchAnalysis analysis = new SearchAnalysis(experiment);
      analysis.setHypervolume(true);
      analysis.setInvertedGenerationalDistance(true);
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

   public Population performSearch(final String initialGraph, final int solutionLength) {
      final TransformationSearchOrchestration orchestration = createOrchestration(initialGraph, solutionLength);
      deriveBaseName(orchestration);
      printSearchInfo(orchestration);
      final SearchExperiment<TransformationSolution> experiment = createExperiment(orchestration);
      experiment.run();
      final NondominatedPopulation population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      return population;
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
      System.out.println("Iterations:      " + maxEvaluations / populationSize);
      System.out.println("MaxEvaluations:  " + maxEvaluations);
      System.out.println("AlgorithmRuns:   " + nrRuns);
      System.out.println("---------------------------");
   }
}
