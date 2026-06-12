package icmt.tool.momot.demo;

import at.ac.tuwien.big.moea.SearchAnalysis;
import at.ac.tuwien.big.moea.SearchExperiment;
import at.ac.tuwien.big.moea.experiment.analyzer.SearchAnalyzer;
import at.ac.tuwien.big.moea.experiment.analyzer.kneepoint.PopulationAnalyzer;
import at.ac.tuwien.big.moea.experiment.executor.listener.SeedRuntimePrintListener;
import at.ac.tuwien.big.moea.print.IPopulationWriter;
import at.ac.tuwien.big.moea.print.ISolutionWriter;
import at.ac.tuwien.big.moea.search.algorithm.EvolutionaryAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.LocalSearchAlgorithmFactory;
import at.ac.tuwien.big.moea.search.algorithm.provider.IRegisteredAlgorithm;
import at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension;
import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.TransformationResultManager;
import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.fitness.EGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.dimension.AbstractEGraphFitnessDimension;
import at.ac.tuwien.big.momot.search.fitness.dimension.TransformationLengthDimension;
import at.ac.tuwien.big.momot.util.MomotUtil;
import com.google.common.base.Objects;
import icmt.tool.momot.demo.architecture.ArchitectureFactory;
import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.ClassModel;
import icmt.tool.momot.demo.architecture.Feature;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
import org.eclipse.xtext.xbase.lib.CollectionExtensions;
import org.eclipse.xtext.xbase.lib.ExclusiveRange;
import org.eclipse.xtext.xbase.lib.Functions.Function1;
import org.eclipse.xtext.xbase.lib.IterableExtensions;
import org.moeaframework.algorithm.NSGAII;
import org.moeaframework.core.Population;
import org.moeaframework.util.progress.ProgressListener;

@SuppressWarnings("all")
public class ArchitectureSearch {
  protected static final String INITIAL_MODEL = "problem/Cart_Item.xmi";

  protected static final int SOLUTION_LENGTH = 80;

  protected final String[] modules = new String[] { "transformations/architecture.henshin" };

  protected final int populationSize = 100;

  protected final int maxEvaluations = 10000;

  protected final int nrRuns = 10;

  protected String baseName;

  protected double _createObjectiveHelper_0(final TransformationSolution solution, final EGraph graph, final EObject root) {
    return FitnessCalculator.calculateCRAIndex(((ClassModel) root));
  }

  protected IFitnessDimension<TransformationSolution> _createObjective_0(final TransformationSearchOrchestration orchestration) {
    return new AbstractEGraphFitnessDimension("CRAIndex", at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Maximum) {
       @Override
       protected double internalEvaluate(TransformationSolution solution) {
          EGraph graph = solution.execute();
          EObject root = MomotUtil.getRoot(graph);
          return _createObjectiveHelper_0(solution, graph, root);
       }
    };
  }

  protected IFitnessDimension<TransformationSolution> _createObjectiveHelper_1() {
    TransformationLengthDimension _transformationLengthDimension = new TransformationLengthDimension();
    return _transformationLengthDimension;
  }

  protected IFitnessDimension<TransformationSolution> _createObjective_1(final TransformationSearchOrchestration orchestration) {
    IFitnessDimension<TransformationSolution> dimension = _createObjectiveHelper_1();
    dimension.setName("SolutionLength");
    dimension.setFunctionType(at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum);
    return dimension;
  }

  protected ModuleManager createModuleManager() {
    ModuleManager manager = new ModuleManager();
    for(String module : modules) {
       manager.addModule(URI.createFileURI(new File(module).getPath().toString()).toString());
    }
    return manager;
  }

  protected IEGraphMultiDimensionalFitnessFunction createFitnessFunction(final TransformationSearchOrchestration orchestration) {
    IEGraphMultiDimensionalFitnessFunction function = new EGraphMultiDimensionalFitnessFunction();
    function.addObjective(_createObjective_0(orchestration));
    function.addObjective(_createObjective_1(orchestration));
    return function;
  }

  protected IRegisteredAlgorithm<NSGAII> _createRegisteredAlgorithm_0(final TransformationSearchOrchestration orchestration, final EvolutionaryAlgorithmFactory<TransformationSolution> moea, final LocalSearchAlgorithmFactory<TransformationSolution> local) {
    IRegisteredAlgorithm<NSGAII> _createNSGAII = moea.createNSGAII();
    return _createNSGAII;
  }

  protected ProgressListener _createListener_0() {
    SeedRuntimePrintListener _seedRuntimePrintListener = new SeedRuntimePrintListener();
    return _seedRuntimePrintListener;
  }

  protected EGraph createInputGraph(final String initialGraph, final ModuleManager moduleManager) {
    EGraph graph = moduleManager.loadGraph(initialGraph);
    return adaptInputGraph(moduleManager, graph);
  }

  protected EGraph adaptInputGraph(final ModuleManager moduleManager, final EGraph initialGraph) {
    EGraph problemGraph = MomotUtil.copy(initialGraph);
    EObject root = MomotUtil.getRoot(problemGraph);
    return MomotUtil.createEGraph(adaptInputModel(root));
  }

  protected EObject adaptInputModel(final EObject root) {
    ClassModel cm = ((ClassModel) root);
    int _size = cm.getFeatures().size();
    int _size_1 = cm.getClasses().size();
    int _minus = (_size - _size_1);
    ExclusiveRange _doubleDotLessThan = new ExclusiveRange(0, _minus, true);
    for (final Integer i : _doubleDotLessThan) {
      {
        icmt.tool.momot.demo.architecture.Class newClass = ArchitectureFactory.eINSTANCE.createClass();
        newClass.setName(("Class_" + i));
        cm.getClasses().add(newClass);
      }
    }
    int i_1 = 0;
    EList<Feature> _features = cm.getFeatures();
    for (final Feature feature : _features) {
      icmt.tool.momot.demo.architecture.Class _isEncapsulatedBy = feature.getIsEncapsulatedBy();
      boolean _equals = Objects.equal(_isEncapsulatedBy, null);
      if (_equals) {
        EList<icmt.tool.momot.demo.architecture.Class> _classes = cm.getClasses();
        int _plusPlus = i_1++;
        _classes.get(_plusPlus).getEncapsulates().add(feature);
      }
    }
    return cm;
  }

  protected TransformationSearchOrchestration createOrchestration(final String initialGraph, final int solutionLength) {
    TransformationSearchOrchestration orchestration = new TransformationSearchOrchestration();
    ModuleManager moduleManager = createModuleManager();
    EGraph graph = createInputGraph(initialGraph, moduleManager);
    orchestration.setModuleManager(moduleManager);
    orchestration.setProblemGraph(graph);
    orchestration.setSolutionLength(solutionLength);
    orchestration.setFitnessFunction(createFitnessFunction(orchestration));
    
    EvolutionaryAlgorithmFactory<TransformationSolution> moea = orchestration.createEvolutionaryAlgorithmFactory(populationSize);
    LocalSearchAlgorithmFactory<TransformationSolution> local = orchestration.createLocalSearchAlgorithmFactory();
    orchestration.addAlgorithm("NSGAII", _createRegisteredAlgorithm_0(orchestration, moea, local));
    
    return orchestration;
  }

  protected SearchExperiment<TransformationSolution> createExperiment(final TransformationSearchOrchestration orchestration) {
    SearchExperiment<TransformationSolution> experiment = new SearchExperiment<TransformationSolution>(orchestration, maxEvaluations);
    experiment.setNumberOfRuns(nrRuns);
    experiment.addProgressListener(_createListener_0());
    return experiment;
  }

  protected void deriveBaseName(final TransformationSearchOrchestration orchestration) {
    EObject root = MomotUtil.getRoot(orchestration.getProblemGraph());
    if(root == null || root.eResource() == null || root.eResource().getURI() == null)
    	baseName = getClass().getSimpleName();
    else
    	baseName = root.eResource().getURI().trimFileExtension().lastSegment();
  }

  protected double significanceLevel = 0.01;

  protected SearchAnalyzer performAnalysis(final SearchExperiment<TransformationSolution> experiment) {
    SearchAnalysis analysis = new SearchAnalysis(experiment);
    analysis.setHypervolume(true);
    analysis.setInvertedGenerationalDistance(true);
    analysis.setShowAggregate(true);
    analysis.setShowIndividualValues(true);
    analysis.setShowStatisticalSignificance(true);
    analysis.setSignificanceLevel(significanceLevel);
    SearchAnalyzer searchAnalyzer = analysis.analyze();
    System.out.println("---------------------------");
    System.out.println("Analysis Results");
    System.out.println("---------------------------");
    searchAnalyzer.printAnalysis();
    System.out.println("---------------------------");
    try {
    	System.out.println("- Save Analysis to 'output/analysis/analysis.txt'");
    	searchAnalyzer.saveAnalysis(new File("output/analysis/analysis.txt"));
    } catch(IOException e) {
    	e.printStackTrace();
    }
    System.out.println("- Save Indicator BoxPlots to 'output/analysis/'");
    searchAnalyzer.saveIndicatorBoxPlots(
    	"output/analysis/",
    	baseName
    );
    return searchAnalyzer;
  }

  protected TransformationResultManager handleResults(final SearchExperiment<TransformationSolution> experiment) {
    ISolutionWriter<TransformationSolution> solutionWriter = experiment.getSearchOrchestration().createSolutionWriter();
    IPopulationWriter<TransformationSolution> populationWriter = experiment.getSearchOrchestration().createPopulationWriter();
    TransformationResultManager resultManager = new TransformationResultManager(experiment);
    Population population;
    population = 
    	TransformationResultManager.createApproximationSet(experiment, (String[])null);
    System.out.println("- Save objectives of all algorithms to 'output/objectives/objective_values.txt'");
    TransformationResultManager.saveObjectives(
    	"output/objectives/objective_values.txt",
    	population
    );
    System.out.println("---------------------------");
    System.out.println("Objectives of all algorithms");
    System.out.println("---------------------------");
    System.out.println(TransformationResultManager.printObjectives(
    	population
    ));
    
    population = 
    	TransformationResultManager.createApproximationSet(experiment, "NSGAII");
    System.out.println("- Save objectives of 'NSGAII' to 'output/objectives/random_objective_values.txt'");
    TransformationResultManager.saveObjectives(
    	"output/objectives/random_objective_values.txt",
    	population
    );
    
    population = 
    	TransformationResultManager.createApproximationSet(experiment, (String[])null);
    System.out.println("- Save solutions of all algorithms to 'output/solutions/all_solutions.txt'");
    TransformationResultManager.savePopulation(
    	"output/solutions/all_solutions.txt",
    	population,
    	populationWriter
    );
    System.out.println("- Save solutions of all algorithms to 'output/solutions/all_solutions.txt'");
    TransformationResultManager.saveSolutions(
    	"output/solutions/",
    	baseName,
    	MomotUtil.asIterables(
    		population,
    		TransformationSolution.class),
    	solutionWriter
    );
    
    population = 
    	TransformationResultManager.createApproximationSet(experiment, (String[])null);
    System.out.println("- Save models of all algorithms to 'output/models/'");
    adaptResultModels(TransformationResultManager.saveModels(
    	"output/models/",
    	baseName,
    	population
    ));
    
    population = 
    	new PopulationAnalyzer(TransformationResultManager.createApproximationSet(experiment))
    		.getKneePointSolutions(
    			TransformationResultManager.createApproximationSet(experiment, (String[])null),
    			populationSize-1,
    			PopulationAnalyzer.FitnessComparison.PROPER_UTILITY);
    System.out.println("- Save Kneepoint models of all algorithms with maximum neighbors to 'output/models/kneepoints/'");
    adaptResultModels(TransformationResultManager.saveModels(
    	"output/models/kneepoints/",
    	baseName,
    	population
    ));
    
    return resultManager;
  }

  protected void adaptResultModels(final List<File> modelFiles) {
    HenshinResourceSet set = new HenshinResourceSet();
    for(File file : modelFiles) {
    	EGraph graph = MomotUtil.loadGraph(file.getPath());
    	EObject root = MomotUtil.getRoot(graph);
    	adaptResultModel(root);
    	MomotUtil.saveGraph(graph, file.getPath());
    }
  }

  protected void adaptResultModel(final EObject root) {
    final ClassModel cm = ((ClassModel) root);
    final Function1<icmt.tool.momot.demo.architecture.Class, Boolean> _function = (icmt.tool.momot.demo.architecture.Class c) -> {
      int _size = c.getEncapsulates().size();
      return Boolean.valueOf((_size == 0));
    };
    final Iterable<icmt.tool.momot.demo.architecture.Class> emptyClasses = IterableExtensions.<icmt.tool.momot.demo.architecture.Class>filter(cm.getClasses(), _function);
    CollectionExtensions.<icmt.tool.momot.demo.architecture.Class>removeAll(cm.getClasses(), emptyClasses);
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

  public void performSearch(final String initialGraph, final int solutionLength) {
    TransformationSearchOrchestration orchestration = createOrchestration(initialGraph, solutionLength);
    deriveBaseName(orchestration);
    printSearchInfo(orchestration);
    SearchExperiment<TransformationSolution> experiment = createExperiment(orchestration);
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

  public static void initialization() {
    ArchitecturePackage.eINSTANCE.eClass();
    System.out.println("Search started.");
  }

  public static void finalization() {
    System.out.println("Search finished.");
  }

  public static void main(final String... args) {
    initialization();
    ArchitectureSearch search = new ArchitectureSearch();
    search.performSearch(INITIAL_MODEL, SOLUTION_LENGTH);
    finalization();
  }
}
