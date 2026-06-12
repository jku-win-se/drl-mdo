package sprintplanning;

import at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension;
import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.learn.NNDelegate;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies.SprintPlanningEncodingStrategy;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.BaseEnvironment;
import at.ac.tuwien.big.momot.search.fitness.EGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.dimension.AbstractEGraphFitnessDimension;
import at.ac.tuwien.big.momot.search.fitness.dimension.TransformationLengthDimension;
import at.ac.tuwien.big.momot.search.solution.repair.TransformationPlaceholderRepairer;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.henshin.interpreter.EGraph;

import SprintPlanning.Plan;
import SprintPlanning.Sprint;
import SprintPlanning.SprintPlanningFactory;
import SprintPlanning.SprintPlanningPackage;
import SprintPlanning.WorkItem;
import co.gongzh.procbridge.Server;

public class SprintLearning {
   private static final String TEST_MODELS = "problem/nn";

   public static IFitnessDimension<TransformationSolution> _createConstraint_0() {
      return new AbstractEGraphFitnessDimension("constraintCombined",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createConstraintHelper_0(solution, graph, root);
         }
      };
   }

   protected static double _createConstraintHelper_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return sprintplanning.FitnessUtils.constraintViolationCombined((Plan) root);
   }

   public static IFitnessDimension<TransformationSolution> _createObjective_0() {
      return new AbstractEGraphFitnessDimension("avgCombinedNorm",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createObjectiveHelper_0(solution, graph, root);
         }
      };
   }

   protected static IFitnessDimension<TransformationSolution> _createObjective_1() {
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

   protected static double _createObjectiveHelper_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.avgCombinedNorm((Plan) root);
   }

   protected static double _createObjectiveHelper_1(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return MinimiseSprintEffortDeviation.computeFitness((Plan) root);
   }

   public static IFitnessDimension<TransformationSolution> _createObjectiveHelper_2() {
      final TransformationLengthDimension _transformationLengthDimension = new TransformationLengthDimension();
      return _transformationLengthDimension;
   }

   public static IFitnessDimension<TransformationSolution> _createRealConstraint_0() {
      return new AbstractEGraphFitnessDimension("noUnassignedWorkitems",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createRealConstraintHelper_0(solution, graph, root);
         }
      };
   }

   public static IFitnessDimension<TransformationSolution> _createRealConstraint_1() {
      return new AbstractEGraphFitnessDimension("noExceedingMaxNumOfSprints",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createRealConstraintHelper_1(solution, graph, root);
         }
      };
   }

   public static IFitnessDimension<TransformationSolution> _createRealConstraint_2() {
      return new AbstractEGraphFitnessDimension("noFallingBelowMinNumOfSprints",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createRealConstraintHelper_2(solution, graph, root);
         }
      };
   }

   public static IFitnessDimension<TransformationSolution> _createRealConstraint_3() {
      return new AbstractEGraphFitnessDimension("noMultiAssignedWorkItems",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createRealConstraintHelper_3(solution, graph, root);
         }
      };
   }

   protected static double _createRealConstraintHelper_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return sprintplanning.FitnessUtils.noUnassignedWorkitems((Plan) root);
   }

   protected static double _createRealConstraintHelper_1(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.noExceedingMaxNumOfSprints((Plan) root);
   }

   protected static double _createRealConstraintHelper_2(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.noFallingBelowMinNumOfSprints((Plan) root);
   }

   protected static double _createRealConstraintHelper_3(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.noMultiAssignedWorkItems((Plan) root);
   }

   public static IFitnessDimension<TransformationSolution> _createRealObjective_0() {
      return new AbstractEGraphFitnessDimension("CustomerImportanceDev",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createRealObjectiveHelper_0(solution, graph, root);
         }
      };
   }

   public static IFitnessDimension<TransformationSolution> _createRealObjective_1() {
      return new AbstractEGraphFitnessDimension("SprintEffortDev",
            at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

         @Override
         protected double internalEvaluate(final TransformationSolution solution) {
            final EGraph graph = solution.execute();
            final EObject root = MomotUtil.getRoot(graph);
            return _createRealObjectiveHelper_1(solution, graph, root);
         }
      };
   }

   public static IFitnessDimension<TransformationSolution> _createRealObjective_2() {
      final IFitnessDimension<TransformationSolution> dimension = _createObjectiveHelper_2();
      dimension.setName("SolutionLength");
      dimension.setFunctionType(at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum);
      return dimension;
   }

   protected static double _createRealObjectiveHelper_0(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return FitnessUtils.computeFitness((Plan) root);
   }

   protected static double _createRealObjectiveHelper_1(final TransformationSolution solution, final EGraph graph,
         final EObject root) {
      return MinimiseSprintEffortDeviation.computeFitness((Plan) root);
   }

   protected static EGraph adaptInputGraph(final ModuleManager moduleManager, final EGraph initialGraph) {
      final EGraph problemGraph = MomotUtil.copy(initialGraph);
      final EObject root = MomotUtil.getRoot(problemGraph);
      return MomotUtil.createEGraph(adaptInputModel(root));
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
      // for(final Sprint sp : p.getSprints()) {
      // for(final WorkItem wi : sp.getCommittedItem()) {
      // System.out.print(wi.getName() + " @ " + sp.getName());
      // System.out.print(", " + wi.getName() + " @ " + wi.getIsPlannedFor().get(0).getName() + " / ");
      // }
      // }
      p.setCurrentSprints(workItemList.size());
      return p;
   }

   public static EGraph createInputGraph(final String initialGraph, final ModuleManager moduleManager) {
      final EGraph graph = moduleManager.loadGraph(initialGraph);
      return adaptInputGraph(moduleManager, graph);
   }

   public static void main(final String[] args) throws IOException {
      SprintPlanningPackage.eINSTANCE.getClass();

      // search orchestration
      final SprintPlanningSearch rs = new SprintPlanningSearch();

      final ModuleManager henshin = new ModuleManager("transformations/", "sprint-mutation2.henshin");

      final TransformationSearchOrchestration orchestration = new TransformationSearchOrchestration(henshin,
            "sprint-planning-model-5-stakeholders-119-items.xmi", 10);

      final IEGraphMultiDimensionalFitnessFunction fitnessFunction = orchestration.getFitnessFunction();

      fitnessFunction.addObjective(_createObjective_0());
      // fitnessFunction.addObjective(_createObjective_1());
      fitnessFunction.addConstraint(_createConstraint_0());

      fitnessFunction.setSolutionRepairer(new TransformationPlaceholderRepairer());

      final IEGraphMultiDimensionalFitnessFunction realFitnessFunction = new EGraphMultiDimensionalFitnessFunction();

      realFitnessFunction.addObjective(_createRealObjective_0());
      // realFitnessFunction.addObjective(_createRealObjective_1());
      realFitnessFunction.addConstraint(_createRealConstraint_0());
      realFitnessFunction.addConstraint(_createRealConstraint_1());
      realFitnessFunction.addConstraint(_createRealConstraint_2());
      realFitnessFunction.addConstraint(_createRealConstraint_3());

      // final Engine engine = new MomotEngine(true, new String());
      // final ModuleManager moduleManager = search.getModuleManager();
      // final IEGraphMultiDimensionalFitnessFunction fitnessFunction = new EGraphMultiDimensionalFitnessFunction();
      // final IEGraphMultiDimensionalFitnessFunction fitnessFunction = tso.getFitnessFunction();

      final File[] files = new File(TEST_MODELS).listFiles();
      final List<File> fileList = Arrays.stream(files).filter(f -> !f.isDirectory()).collect(Collectors.toList());

      final List<TransformationSolution> initialSolutions = MomotUtil.readInitialSolutions(fileList,
            fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size());

      // final BaseEnvironment<TransformationSolution> env = learnEnv.init(search);
      final BaseEnvironment<TransformationSolution> env = MomotUtil.initBaseLearnEnv(henshin, fitnessFunction,
            "avgCombinedNorm", "constraintCombined", null, SprintPlanningEncodingStrategy.class);

      // final Server server = new StandalonePy(env, initialSolutions, fileList);
      final Server server = new Server(8000, new NNDelegate(env, realFitnessFunction, initialSolutions, fileList, 100));
      server.start();
      System.out.println("Test Server is on 8000...");
   }

}
