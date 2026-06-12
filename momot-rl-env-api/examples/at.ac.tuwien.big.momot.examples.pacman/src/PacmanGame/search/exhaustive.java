package PacmanGame.search;

import at.ac.tuwien.big.moea.SearchAnalysis;
import at.ac.tuwien.big.moea.SearchExperiment;
import at.ac.tuwien.big.moea.SearchResultManager;
import at.ac.tuwien.big.moea.experiment.analyzer.SearchAnalyzer;
import at.ac.tuwien.big.moea.experiment.executor.listener.SeedRuntimePrintListener;
import at.ac.tuwien.big.moea.print.IPopulationWriter;
import at.ac.tuwien.big.moea.print.ISolutionWriter;
import at.ac.tuwien.big.moea.search.algorithm.EvolutionaryAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.LocalSearchAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.provider.IRegisteredAlgorithm;
import at.ac.tuwien.big.moea.search.fitness.IMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension;
import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.TransformationResultManager;
import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.TransformationStateExplorer;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.problem.solution.variable.ITransformationVariable;
import at.ac.tuwien.big.momot.problem.solution.variable.UnitApplicationVariable;
import at.ac.tuwien.big.momot.search.fitness.EGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.dimension.AbstractEGraphFitnessDimension;
import at.ac.tuwien.big.momot.search.fitness.dimension.TransformationLengthDimension;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.interpreter.RuleApplication;
import org.eclipse.emf.henshin.model.Parameter;
import org.moeaframework.algorithm.NSGAII;
import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;
import org.moeaframework.util.progress.ProgressListener;

import PacmanGame.Game;
import PacmanGame.GridNode;
import PacmanGame.PacmanGamePackage;
import PacmanGame.PositionableEntity;
import PacmanGame.impl.FoodImpl;
import PacmanGame.impl.GameImpl;
import PacmanGame.impl.GhostImpl;
import PacmanGame.impl.PacmanImpl;

@SuppressWarnings("all")
public class exhaustive {
   protected static final String INITIAL_MODEL = "models/input_6x6.xmi";

   protected static final int SOLUTION_LENGTH = 15;

   public static void finalization() {
      System.out.println("Search finished.");
   }

   public static void initialization() {
      PacmanGamePackage.eINSTANCE.getClass();
      System.out.println("Search started.");
   }

   public static void main(final String... args) {
      initialization();
      final exhaustive search = new exhaustive();
      search.performSearch(INITIAL_MODEL, SOLUTION_LENGTH);
      finalization();
   }

   protected final String[] modules = new String[] { "transformation/play_game.henshin" };

   protected final int populationSize = 300;

   protected final int maxEvaluations = 10000;

   protected final int nrRuns = 1;

   protected String baseName;

   protected double significanceLevel = 0.01;

   protected ProgressListener _createListener_0() {
      final SeedRuntimePrintListener _seedRuntimePrintListener = new SeedRuntimePrintListener();
      return _seedRuntimePrintListener;
   }

   protected IFitnessDimension<TransformationSolution> _createObjective_0(
         final TransformationSearchOrchestration orchestration) {
      return new AbstractEGraphFitnessDimension("Score",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Maximum) {
         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createObjectiveHelper_0(solution, graph, root);
         }
      };
   }

   protected IFitnessDimension<TransformationSolution> _createObjective_1(
         final TransformationSearchOrchestration orchestration) {
      final IFitnessDimension<TransformationSolution> dimension = _createObjectiveHelper_1();
      dimension.setName("SolutionLength");
      dimension.setFunctionType(at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum);
      return dimension;
   }

   protected double _createObjectiveHelper_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return ((Game) root).getScoreboard().getScore();
   }

   protected IFitnessDimension<TransformationSolution> _createObjectiveHelper_1() {
      final TransformationLengthDimension _transformationLengthDimension = new TransformationLengthDimension();
      return _transformationLengthDimension;
   }

   protected IRegisteredAlgorithm<NSGAII> _createRegisteredAlgorithm_0(
         final TransformationSearchOrchestration orchestration,
         final EvolutionaryAlgorithmFactory<TransformationSolution> moea,
         final LocalSearchAlgorithmFactory<TransformationSolution> local) {
      final IRegisteredAlgorithm<NSGAII> _createNSGAII = moea.createNSGAII();
      return _createNSGAII;
   }

   private String chainToStringRepresentation(final TransformationSolution s) {
      final StringBuilder sb = new StringBuilder(getChainNameForString(s));
      sb.append(String.format(" (%s)", Arrays.toString(s.getObjectives())));
      return sb.toString();
   }

   protected SearchExperiment<TransformationSolution> createExperiment(
         final TransformationSearchOrchestration orchestration) {
      final SearchExperiment<TransformationSolution> experiment = new SearchExperiment<>(orchestration, maxEvaluations);
      experiment.setNumberOfRuns(nrRuns);
      experiment.addProgressListener(_createListener_0());
      return experiment;
   }

   protected IEGraphMultiDimensionalFitnessFunction createFitnessFunction(
         final TransformationSearchOrchestration orchestration) {
      final IEGraphMultiDimensionalFitnessFunction function = new EGraphMultiDimensionalFitnessFunction();
      function.addObjective(_createObjective_0(orchestration));
      function.addObjective(_createObjective_1(orchestration));
      return function;
   }

   protected EGraph createInputGraph(final String initialGraph, final ModuleManager moduleManager) {
      final EGraph graph = moduleManager.loadGraph(initialGraph);
      return graph;
   }

   protected ModuleManager createModuleManager() {
      final ModuleManager manager = new ModuleManager();
      for(final String module : modules) {
         manager.addModule(URI.createFileURI(new File(module).getPath().toString()).toString());
      }
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
      orchestration.addAlgorithm("NSGAII", _createRegisteredAlgorithm_0(orchestration, moea, local));

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

   private String getChainNameForString(final TransformationSolution s) {

      final String m = printModel(s).insert(0, "\n").toString();
      return m;
   }

   protected TransformationResultManager handleResults(final SearchExperiment<TransformationSolution> experiment,
         final TransformationStateExplorer exploration) {
      final ISolutionWriter<TransformationSolution> solutionWriter = experiment.getSearchOrchestration()
            .createSolutionWriter();
      final IPopulationWriter<TransformationSolution> populationWriter = experiment.getSearchOrchestration()
            .createPopulationWriter();
      final TransformationResultManager resultManager = new TransformationResultManager(experiment);

      Population population = exploration.getSolutions(true);
      System.out.println("- Save objectives of state exploration to 'output/exploration/objective_values.txt'");
      SearchResultManager.saveObjectives("output/exploration/objective_values.txt", population);
      SearchResultManager.saveSolutions("output/exploration/solutions/", baseName,
            MomotUtil.asIterables(population, TransformationSolution.class), solutionWriter);

      // population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save objectives of all algorithms to 'output/objectives/objective_values.txt'");
      SearchResultManager.saveObjectives("output/objectives/objective_values.txt", population);
      System.out.println("---------------------------");
      System.out.println("Objectives of all algorithms");
      System.out.println("---------------------------");
      System.out.println(SearchResultManager.printObjectives(population));

      // population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save solutions of all algorithms to 'output/solutions/objective_values.txt'");
      SearchResultManager.savePopulation("output/solutions/objective_values.txt", population, populationWriter);
      System.out.println("- Save solutions of all algorithms to 'output/solutions/objective_values.txt'");
      SearchResultManager.saveSolutions("output/solutions/", baseName,
            MomotUtil.asIterables(population, TransformationSolution.class), solutionWriter);

      // TransformationResultManager.saveModels(".", "test_dir", population);

      final StringBuilder algo_solutions = new StringBuilder();
      final List<double[]> solutions = new ArrayList<>();
      final List<String> chainNames = new ArrayList<>();

      // for(final Entry<SearchExecutor, List<NondominatedPopulation>> entry : resultManager.getResults().entrySet()) {

      // System.out.println(entry.getKey().getName());
      //
      // population = SearchResultManager.createApproximationSet(experiment, entry.getKey().getName());
      // System.out.println(SearchResultManager.printObjectives(population) + "\n");
      //
      // population = SearchResultManager.createApproximationSet(experiment, entry.getKey().getName());
      // SearchResultManager.saveObjectives("output/objectives/" + entry.getKey().getName() + ".txt", population);
      //
      // algo_solutions.append("----------------------------------------------------------------------\n");
      // algo_solutions.append(entry.getKey().getName());
      // algo_solutions.append("\n\n----------------------------------------------------------------------");

      population = exploration.getSolutions(false);

      int noSolutions = 1;
      for(final TransformationSolution ts : MomotUtil.asIterables(population, TransformationSolution.class)) {
         algo_solutions.append("\nSolution " + noSolutions++ + ":\n");
         for(final ITransformationVariable tv : ts.getVariables()) {
            if(tv instanceof UnitApplicationVariable) {
               final UnitApplicationVariable uav = (UnitApplicationVariable) tv;
               algo_solutions.append(uav.getAssignment().toString());
               for(final RuleApplication ra : uav.getAppliedRules()) {
                  algo_solutions.append(String.format("\tRule '%s'\n", ra.getUnit().getName()));
                  for(final Parameter p : ra.getResultMatch().getUnit().getParameters()) {
                     final Object o = ra.getResultParameterValue(p.getName());
                     algo_solutions.append(String.format("\t\t- parameter '%s' => '%s'\n", p.getName(), o.toString()));
                  }

               }
            }
         }
         algo_solutions.append("\nObjectives: " + Arrays.toString(ts.getObjectives()));

         solutions.add(ts.getObjectives());

      }
      algo_solutions.append("\n\n");

      // }

      final IMultiDimensionalFitnessFunction f = experiment.getSearchOrchestration().getFitnessFunction();
      final List<String> header = new ArrayList<>(f.getObjectiveNames());

      for(int i = 0; i < header.size(); i++) {
         if(header.get(i).equals("SolutionLength")) {
            header.set(i, "Length");
         }
      }

      BufferedWriter writer = null;
      try {
         writer = new BufferedWriter(new FileWriter("output/solutions/units_and_rules.txt", false));
         writer.append(algo_solutions);
         writer.close();
      } catch(final IOException e) {
         e.printStackTrace();
      }

      // population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
      System.out.println("- Save models of all algorithms to 'output/models/'");
      TransformationResultManager.saveModels("output/models/", baseName, population);

      return resultManager;
   }

   // protected TransformationResultManager handleResults(final SearchExperiment<TransformationSolution> experiment) {
   // final ISolutionWriter<TransformationSolution> solutionWriter = experiment.getSearchOrchestration()
   // .createSolutionWriter();
   // final IPopulationWriter<TransformationSolution> populationWriter = experiment.getSearchOrchestration()
   // .createPopulationWriter();
   // final TransformationResultManager resultManager = new TransformationResultManager(experiment);
   // Population population;
   // population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
   // System.out.println("- Save objectives of all algorithms to 'output/objectives/objective_values.txt'");
   // SearchResultManager.saveObjectives("output/objectives/objective_values.txt", population);
   // System.out.println("---------------------------");
   // System.out.println("Objectives of all algorithms");
   // System.out.println("---------------------------");
   // System.out.println(SearchResultManager.printObjectives(population));
   //
   // population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
   // System.out.println("- Save solutions of all algorithms to 'output/solutions/objective_values.txt'");
   // SearchResultManager.savePopulation("output/solutions/objective_values.txt", population, populationWriter);
   // System.out.println("- Save solutions of all algorithms to 'output/solutions/objective_values.txt'");
   // SearchResultManager.saveSolutions("output/solutions/", baseName,
   // MomotUtil.asIterables(population, TransformationSolution.class), solutionWriter);
   //
   // population = SearchResultManager.createApproximationSet(experiment, (String[]) null);
   // System.out.println("- Save models of all algorithms to 'output/models/'");
   // TransformationResultManager.saveModels("output/models/", baseName, population);
   //
   // return resultManager;
   // }

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
      TransformationStateExplorer exploration = null;
      exploration = createStateExploration(INITIAL_MODEL, orchestration.getModuleManager(),
            orchestration.getFitnessFunction());
      exploration.setSolutionReprFunction(s -> chainToStringRepresentation(s));
      exploration.run();

      final SearchExperiment<TransformationSolution> experiment = createExperiment(orchestration);
      experiment.run();

      System.out.println("-------------------------------------------------------");
      System.out.println("Analysis");
      System.out.println("-------------------------------------------------------");
      // performAnalysis(experiment);
      System.out.println("-------------------------------------------------------");
      System.out.println("Results");
      System.out.println("-------------------------------------------------------");
      handleResults(experiment, exploration);
   }

   protected StringBuilder printModel(final Solution s) {
      final GameImpl game = MomotUtil.getRoot(((TransformationSolution) s).getResultGraph(), GameImpl.class);
      final Map<String, String> fieldToChar = new HashMap<>();

      final Map<String, String> nodeStates = new HashMap<>();
      for(final GridNode g : game.getGridnodes()) {
         nodeStates.put(g.getId(), "-");

      }

      for(final PositionableEntity entity : game.getEntites()) {
         if(entity instanceof PacmanImpl) {
            final String pacId = ((PacmanImpl) entity).getOn().getId();
            nodeStates.put(pacId, "P");

         } else if(entity instanceof FoodImpl) {
            final String foodId = ((FoodImpl) entity).getOn().getId();

            nodeStates.put(foodId, "F");

         } else if(entity instanceof GhostImpl) {
            final String ghostId = ((GhostImpl) entity).getOn().getId();

            nodeStates.put(ghostId, "G");

         }
      }

      final List<String> sortedNodeIds = new ArrayList<>(nodeStates.keySet());

      final List<Entry<String, String>> sortedList = nodeStates.entrySet().stream().sorted(Map.Entry.comparingByKey())
            .collect(Collectors.toList());

      int cnt = 1;

      final StringBuilder sb = new StringBuilder();
      for(final Entry<String, String> e : sortedList) {
         sb.append(e.getValue());
         if(cnt % Math.sqrt(game.getGridnodes().size()) == 0) {
            sb.append("\n");
         }
         cnt++;
      }
      return sb;
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
