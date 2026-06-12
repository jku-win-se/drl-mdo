package learnapi;

import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.examples.stack.StackOrchestration;
import at.ac.tuwien.big.momot.examples.stack.stack.StackModel;
import at.ac.tuwien.big.momot.examples.stack.stack.StackPackage;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies.CRAEncodingStrategy;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies.SprintPlanningEncodingStrategy;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies.StackEncodingStrategy;
import at.ac.tuwien.big.momot.search.fitness.EGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.fitness.dimension.AbstractEGraphFitnessDimension;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.ToDoubleFunction;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
import org.json.JSONObject;

import SprintPlanning.Plan;
import SprintPlanning.SprintPlanningPackage;
import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.ClassModel;
import learnapi.ConditionUtils.LimitType;
import sprintplanning.SprintPlanningSearch;

public enum UseCase {
   SLB("Standard Deviation"), CRA("CRAIndex"), SCRUM("avgCombinedNorm", "constraintCombined");

   public static class EnvConfig {
      public IEGraphMultiDimensionalFitnessFunction function;
      public ModuleManager moduleManager;
      public Class encodingStrategyClass;
      public ToDoubleFunction<double[]> fitnessAggregator;
      public ToDoubleFunction<double[]> constraintAggregator;
      public ToDoubleFunction<EObject> fitnessScaler;
      public ToDoubleFunction<EObject> constraintScaler;
   }

   /**
    * Calculate the conditions for a given model based on case-specific logic.
    * Returns null if the case does not use conditions.
    * This is a placeholder implementation - logic will be specified later.
    *
    * @param caseId
    *           The case ID (cra, slb, scrum)
    * @param graph
    *           The input graph to calculate conditions for
    * @return JSONObject with conditions for this model, or null if not applicable
    */
   public static JSONObject calculateConditions(final String caseId, final EGraph graph) {
      switch(caseId.toLowerCase()) {
         case "cra":
            // TODO: Specify conditions logic for CRA case
            // Placeholder: no conditions
            return null;

         case "slb":
            // TODO: Specify conditions logic for SLB (Stack) case
            // Placeholder: no conditions
            return null;

         case "scrum":
            final Plan p = (Plan) graph.getRoots().get(0);
            final int items = p.getBacklog().getWorkitems().size();

            return new ConditionUtils().addCondition(0, "elements", "Sprint", LimitType.MAX, p.getMaxSprints())
                  .addCondition(2, "elements", "Sprint", LimitType.MIN, p.getMinSprints())
                  .addCondition(4, "elements", "Sprint", LimitType.MAX, p.getMaxSprints()).build();
         case "scrum-uc":
            final Plan p2 = (Plan) graph.getRoots().get(0);
            final int items2 = p2.getBacklog().getWorkitems().size();

            return new ConditionUtils().addCondition(0, "elements", "Sprint", LimitType.MAX, p2.getMaxSprints())
                  .addCondition(2, "elements", "Sprint", LimitType.MIN, p2.getMinSprints())
                  .addCondition(4, "elements", "Sprint", LimitType.MAX, p2.getMaxSprints())// .build();
                  .addCondition(1, "relations", "isPlannedFor", LimitType.MIN, items2 - 1)
                  .addCondition(2, "relations", "isPlannedFor", LimitType.MIN, items2 - 1)
                  .addCondition(4, "relations", "isPlannedFor", LimitType.MIN, items2 - 1).build();
         default:
            // Default: no conditions
            return null;
      }
   }

   /**
    * Calculate the number of steps for a given model based on case-specific formula.
    * This is a placeholder implementation - formulas will be specified later.
    *
    * @param caseId
    *           The case ID (cra, slb, scrum)
    * @param graph
    *           The input graph to calculate steps for
    * @return Number of steps for this model
    */
   public static int calculateSteps(final String caseId, final EGraph graph) {
      // Get the number of elements in the graph as a base metric
      final int elementCount = graph.size();

      switch(caseId.toLowerCase()) {
         case "cra":
            final ClassModel cm = (ClassModel) graph.getRoots().get(0);
            return cm.getFeatures().size() * 2;

         case "slb":
            final StackModel sm = (StackModel) graph.getRoots().get(0);

            return (int) Math.floor(sm.getStacks().size() * 2.5);

         case "scrum":
         case "scrum-uc":
            final Plan p = (Plan) graph.getRoots().get(0);

            return (int) Math.floor(p.getBacklog().getWorkitems().size() * 2.3);

         default:
            // Default fallback
            return Math.max(50, elementCount * 2);
      }
   }

   public static double[] constraintNormalizer(final String caseId, final EGraph graph) {
      final double[] c;
      switch(caseId.toLowerCase()) {
         case "cra":
            return null;
         case "slb":
            return null;
         case "scrum":
         case "scrum-uc":
            return null;
         default:
            return null;
      }
   }

   /**
    * Creates a fresh per-environment configuration with its own fitness function
    * and module manager to avoid shared state across environments.
    *
    * @param id
    *           The case ID (cra, slb, scrum)
    * @param modelPath
    *           Full path to a model file (used by cases that require a model in orchestration)
    * @return Environment configuration with fresh instances
    */
   public static EnvConfig createEnvConfig(final String id, final String modelPath) {
      final EnvConfig config = new EnvConfig();

      switch(id.toLowerCase()) {
         case "cra":
            ArchitecturePackage.eINSTANCE.eClass();
            final TransformationSearchOrchestration tso = icmt.tool.momot.demo.SOArchitectureSearch
                  .createOrchestration(modelPath, 1);
            config.function = tso.getFitnessFunction();
            config.fitnessAggregator = (final double[] objectiveVals) -> {
               return objectiveVals[0] + objectiveVals[1];
               // return objectiveVals[0];
            };
            config.fitnessScaler = (final EObject m) -> {
               return 1000.;

            };
            config.moduleManager = tso.getModuleManager();
            config.encodingStrategyClass = CRAEncodingStrategy.class;
            break;

         case "slb":
            StackPackage.eINSTANCE.eClass();
            final TransformationSearchOrchestration tsoSLB = new StackOrchestration(modelPath, 1);
            config.function = tsoSLB.getFitnessFunction();
            config.fitnessAggregator = (final double[] objectiveVals) -> {
               return objectiveVals[0];
            };
            config.moduleManager = tsoSLB.getModuleManager();
            config.encodingStrategyClass = StackEncodingStrategy.class;
            break;

         case "scrum":
            SprintPlanningPackage.eINSTANCE.getClass();
            final IEGraphMultiDimensionalFitnessFunction fitnessFunction = new EGraphMultiDimensionalFitnessFunction();
            fitnessFunction.addConstraint(SprintPlanningSearch._createConstraint_rl_0());
            fitnessFunction.addObjective(SprintPlanningSearch._createObjective_0());
            fitnessFunction.addObjective(SprintPlanningSearch._createObjective_1());
            fitnessFunction.addObjective(SprintPlanningSearch._createObjective_2());
            config.function = fitnessFunction;

            config.fitnessAggregator = (final double[] objectiveVals) -> {
               return objectiveVals[0] + objectiveVals[1];
            };

            config.constraintAggregator = (final double[] constraintVals) -> {
               return constraintVals[0];
            };

            config.moduleManager = createModuleManager(new String[] { "transformations/sprint-mutation2.henshin" },
                  new String[] { "sprint-mutation::moveWorkItemToNewSprint", "sprint-mutation::deleteEmptySprint",
                        "sprint-mutation::addUnassignedWorkItemToNewSprint" });
            config.encodingStrategyClass = SprintPlanningEncodingStrategy.class;
            break;

         case "scrum-uc":
            SprintPlanningPackage.eINSTANCE.getClass();

            final IEGraphMultiDimensionalFitnessFunction fitnessFunctionUC = new EGraphMultiDimensionalFitnessFunction();
            fitnessFunctionUC.addConstraint(new AbstractEGraphFitnessDimension("constraintCombined",
                  at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

               @Override
               protected double internalEvaluate(final TransformationSolution solution) {
                  final EGraph graph = solution.execute();
                  final EObject root = MomotUtil.getRoot(graph);
                  return sprintplanning.FitnessUtils.constraintViolationCombined((Plan) root);
               }
            });
            fitnessFunctionUC.addObjective(SprintPlanningSearch._createObjective_0());
            fitnessFunctionUC.addObjective(SprintPlanningSearch._createObjective_1());
            fitnessFunctionUC.addObjective(SprintPlanningSearch._createObjective_2());
            config.function = fitnessFunctionUC;

            config.fitnessAggregator = (final double[] objectiveVals) -> {
               return objectiveVals[0] + objectiveVals[1];
            };

            config.moduleManager = createModuleManager(new String[] { "transformations/sprint-mutation2.henshin" },
                  new String[] { "sprint-mutation::moveWorkItemToNewSprint", "sprint-mutation::deleteEmptySprint",
                        "sprint-mutation::addUnassignedWorkItemToNewSprint" });
            config.encodingStrategyClass = SprintPlanningEncodingStrategy.class;
            break;

         default:
            throw new IllegalArgumentException("Unknown use case: " + id);
      }

      return config;
   }

   protected static ModuleManager createModuleManager(final String[] modules, final String[] unitsToRemove) {
      final ModuleManager manager = new ModuleManager();
      for(final String module : modules) {
         manager.addModule(URI.createFileURI(new File(module).getPath().toString()).toString());
      }
      manager.removeUnits(unitsToRemove);

      return manager;
   }

   public static UseCase fromId(final String id, final String inputModel) {
      UseCase uc = null;
      final HenshinResourceSet henshinResourceSet = new HenshinResourceSet();
      Resource modelResource = null;
      switch(id.toLowerCase()) {
         case "cra":
            uc = CRA;
            ArchitecturePackage.eINSTANCE.eClass();
            final TransformationSearchOrchestration tso = icmt.tool.momot.demo.SOArchitectureSearch
                  .createOrchestration(Path.of(inputModel + ".xmi").toString(), 1);

            // final TransformationSearchOrchestration tso = ArchitectureSearcher
            // .createOrchestration(Path.of("..", "org.cra2", "problem", inputModel + ".xmi").toString(), 1);

            uc.function = tso.getFitnessFunction();
            uc.fitnessAggregator = (final double[] objectiveVals) -> {
               // return objectiveVals[0] + objectiveVals[1];
               return objectiveVals[0];

            };
            uc.realFunction = uc.function;
            uc.moduleManager = tso.getModuleManager();
            uc.encodingStrategyClass = CRAEncodingStrategy.class;
            uc.inputModel = Path.of(inputModel + ".xmi").toString();
            uc.testModelPath = Path.of(".");
            uc.inputGraph = icmt.tool.momot.demo.architecture.ArchitectureSearcher.createInputGraph(uc.inputModel,
                  tso.getModuleManager());
            break;
         case "slb":
            uc = SLB;
            StackPackage.eINSTANCE.eClass();

            final TransformationSearchOrchestration tsoSLB = new StackOrchestration(
                  Path.of(inputModel + ".xmi").toString(), 1);
            uc.function = tsoSLB.getFitnessFunction();
            uc.fitnessAggregator = (final double[] objectiveVals) -> {
               return objectiveVals[0];
            };
            uc.realFunction = uc.function;
            uc.moduleManager = tsoSLB.getModuleManager();
            uc.encodingStrategyClass = StackEncodingStrategy.class;
            uc.inputModel = Path.of(inputModel + ".xmi").toString();
            uc.testModelPath = Path.of(".");

            modelResource = henshinResourceSet.getResource(uc.inputModel);
            uc.inputGraph = MomotUtil.eGraphOf(modelResource, true);
            break;

         case "scrum":
            uc = SCRUM;
            SprintPlanningPackage.eINSTANCE.getClass();

            final IEGraphMultiDimensionalFitnessFunction fitnessFunction = new EGraphMultiDimensionalFitnessFunction();

            fitnessFunction.addConstraint(SprintPlanningSearch._createConstraint_rl_0());
            fitnessFunction.addObjective(SprintPlanningSearch._createObjective_0());

            fitnessFunction.addObjective(SprintPlanningSearch._createObjective_1());
            fitnessFunction.addObjective(SprintPlanningSearch._createObjective_2());

            uc.function = fitnessFunction;
            uc.fitnessAggregator = (final double[] objectiveVals) -> {
               return objectiveVals[0] + objectiveVals[1];
            };

            uc.constraintAggregator = (final double[] constraintVals) -> {
               return constraintVals[0];
            };

            uc.moduleManager = createModuleManager(new String[] { "transformations/sprint-mutation2.henshin" },
                  new String[] { "sprint-mutation::moveWorkItemToNewSprint", "sprint-mutation::deleteEmptySprint",
                        "sprint-mutation::addUnassignedWorkItemToNewSprint" });

            uc.encodingStrategyClass = SprintPlanningEncodingStrategy.class;
            uc.inputModel = Path.of("model", inputModel + ".xmi").toString();
            uc.testModelPath = Path.of("model");

            modelResource = henshinResourceSet.getResource(uc.inputModel);
            uc.inputGraph = MomotUtil.eGraphOf(modelResource, true);

            break;
         case "scrum-uc":
            uc = SCRUM;
            SprintPlanningPackage.eINSTANCE.getClass();

            final IEGraphMultiDimensionalFitnessFunction fitnessFunctionUC = new EGraphMultiDimensionalFitnessFunction();

            fitnessFunctionUC.addConstraint(new AbstractEGraphFitnessDimension("constraintCombined",
                  at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType.Minimum) {

               @Override
               protected double internalEvaluate(final TransformationSolution solution) {
                  final EGraph graph = solution.execute();
                  final EObject root = MomotUtil.getRoot(graph);
                  return sprintplanning.FitnessUtils.constraintViolationCombined((Plan) root);
               }
            });
            fitnessFunctionUC.addObjective(SprintPlanningSearch._createObjective_0());
            fitnessFunctionUC.addObjective(SprintPlanningSearch._createObjective_1());
            fitnessFunctionUC.addObjective(SprintPlanningSearch._createObjective_2());
            uc.function = fitnessFunctionUC;

            uc.fitnessAggregator = (final double[] objectiveVals) -> {
               return objectiveVals[0] + objectiveVals[1];
            };

            uc.moduleManager = createModuleManager(new String[] { "transformations/sprint-mutation2.henshin" },
                  new String[] { "sprint-mutation::moveWorkItemToNewSprint", "sprint-mutation::deleteEmptySprint",
                        "sprint-mutation::addUnassignedWorkItemToNewSprint" });

            uc.encodingStrategyClass = SprintPlanningEncodingStrategy.class;
            uc.inputModel = Path.of(inputModel + ".xmi").toString();
            uc.testModelPath = Path.of(".");

            modelResource = henshinResourceSet.getResource(uc.inputModel);
            uc.inputGraph = MomotUtil.eGraphOf(modelResource, true);

            break;
         default:
            throw new IllegalArgumentException("Unknown use case: " + id);
      }

      return uc;
   }

   public static UseCase fromId(final String id, final String inputModel, final JSONObject conditions) {
      final UseCase uc = UseCase.fromId(id, inputModel);
      uc.conditions = conditions;
      return uc;
   }

   /**
    * Factory method for multi-model setup.
    * Loads all .xmi models from the given folder path and pre-calculates steps for each.
    *
    * @param id
    *           The case ID (cra, slb, scrum)
    * @param modelsPath
    *           Path to folder containing .xmi model files
    * @return UseCase configured with multiple input graphs and their step counts
    */
   public static UseCase fromIdMultiModel(final String id, final String modelsPath) {
      // First, get base UseCase configuration using first model found
      final File folder = new File(modelsPath);
      final File[] xmiFiles = folder.listFiles((dir, name) -> name.endsWith(".xmi"));

      if(xmiFiles == null || xmiFiles.length == 0) {
         throw new IllegalArgumentException("No .xmi files found in: " + modelsPath);
      }

      // Sort files for deterministic ordering
      Arrays.sort(xmiFiles);

      // Get first model name (without .xmi extension) to initialize base UseCase
      final String firstModelName = xmiFiles[0].getName().replace(".xmi", "");

      // Initialize UseCase with first model to get moduleManager, function, etc.
      final UseCase uc;
      final HenshinResourceSet henshinResourceSet = new HenshinResourceSet();

      switch(id.toLowerCase()) {
         case "cra":
            uc = CRA;
            ArchitecturePackage.eINSTANCE.eClass();
            final TransformationSearchOrchestration tso = icmt.tool.momot.demo.SOArchitectureSearch
                  .createOrchestration(xmiFiles[0].getPath(), 1);
            uc.function = tso.getFitnessFunction();
            uc.fitnessAggregator = (final double[] objectiveVals) -> {
               // return objectiveVals[0] + objectiveVals[1];
               return objectiveVals[0];

            };
            uc.realFunction = uc.function;
            uc.moduleManager = tso.getModuleManager();
            uc.encodingStrategyClass = CRAEncodingStrategy.class;
            uc.testModelPath = Path.of(modelsPath);
            break;

         case "slb":
            uc = SLB;
            StackPackage.eINSTANCE.eClass();
            final TransformationSearchOrchestration tsoSLB = new StackOrchestration(xmiFiles[0].getPath(), 1);
            uc.function = tsoSLB.getFitnessFunction();
            uc.fitnessAggregator = (final double[] objectiveVals) -> {
               return objectiveVals[0];
            };
            uc.realFunction = uc.function;
            uc.moduleManager = tsoSLB.getModuleManager();
            uc.encodingStrategyClass = StackEncodingStrategy.class;
            uc.testModelPath = Path.of(modelsPath);
            break;

         case "scrum":
            uc = SCRUM;
            SprintPlanningPackage.eINSTANCE.getClass();
            final IEGraphMultiDimensionalFitnessFunction fitnessFunction = new EGraphMultiDimensionalFitnessFunction();
            fitnessFunction.addConstraint(SprintPlanningSearch._createConstraint_rl_0());
            fitnessFunction.addObjective(SprintPlanningSearch._createObjective_0());
            fitnessFunction.addObjective(SprintPlanningSearch._createObjective_1());
            fitnessFunction.addObjective(SprintPlanningSearch._createObjective_2());
            uc.function = fitnessFunction;

            uc.fitnessAggregator = (final double[] objectiveVals) -> {
               return objectiveVals[0] + objectiveVals[1];
            };

            uc.constraintAggregator = (final double[] constraintVals) -> {
               return constraintVals[0];
            };

            uc.moduleManager = createModuleManager(new String[] { "transformations/sprint-mutation2.henshin" },
                  new String[] { "sprint-mutation::moveWorkItemToNewSprint", "sprint-mutation::deleteEmptySprint",
                        "sprint-mutation::addUnassignedWorkItemToNewSprint" });
            uc.encodingStrategyClass = SprintPlanningEncodingStrategy.class;
            uc.testModelPath = Path.of(modelsPath);
            break;
         case "scrum-uc":
            uc = SCRUM;
            SprintPlanningPackage.eINSTANCE.getClass();
            final IEGraphMultiDimensionalFitnessFunction fitnessFunctionUC = new EGraphMultiDimensionalFitnessFunction();
            fitnessFunctionUC.addConstraint(SprintPlanningSearch._createConstraint_rl_0());
            fitnessFunctionUC.addObjective(SprintPlanningSearch._createObjective_0());
            fitnessFunctionUC.addObjective(SprintPlanningSearch._createObjective_1());
            fitnessFunctionUC.addObjective(SprintPlanningSearch._createObjective_2());
            uc.function = fitnessFunctionUC;

            uc.fitnessAggregator = (final double[] objectiveVals) -> {
               return objectiveVals[0] + objectiveVals[1];
            };
            uc.moduleManager = createModuleManager(new String[] { "transformations/sprint-mutation2.henshin" },
                  new String[] { "sprint-mutation::moveWorkItemToNewSprint", "sprint-mutation::deleteEmptySprint",
                        "sprint-mutation::addUnassignedWorkItemToNewSprint" });
            uc.encodingStrategyClass = SprintPlanningEncodingStrategy.class;
            uc.testModelPath = Path.of(modelsPath);
            break;

         default:
            throw new IllegalArgumentException("Unknown use case: " + id);
      }

      // Now load all models and calculate steps and conditions for each
      uc.inputGraphs = new ArrayList<>();
      uc.modelNames = new ArrayList<>();
      uc.stepsPerModel = new ArrayList<>();
      uc.conditionsPerModel = new ArrayList<>();
      uc.fitnessNormalizersPerModel = new ArrayList<>();
      uc.constraintNormalizersPerModel = new ArrayList<>();

      for(final File xmiFile : xmiFiles) {
         final String modelName = xmiFile.getName().replace(".xmi", "");
         uc.modelNames.add(modelName);

         final Resource modelResource = henshinResourceSet.getResource(xmiFile.getPath());

         final EGraph graph;
         switch(id.toLowerCase()) {
            case "cra":
               final TransformationSearchOrchestration tso = icmt.tool.momot.demo.SOArchitectureSearch
                     .createOrchestration(xmiFiles[0].getPath(), 1);
               graph = icmt.tool.momot.demo.architecture.ArchitectureSearcher.createInputGraph(xmiFile.getPath(),
                     tso.getModuleManager());
               break;
            default:
               graph = MomotUtil.eGraphOf(modelResource, true);
               break;
         }
         uc.inputGraphs.add(graph);

         final int steps = calculateSteps(id, graph);
         uc.stepsPerModel.add(steps);

         final JSONObject conditions = calculateConditions(id, graph);
         uc.conditionsPerModel.add(conditions);

         final double[] fitnessNormalizers = objectiveNormalizer(id, graph);
         uc.fitnessNormalizersPerModel.add(fitnessNormalizers);

         final double[] constraintNormalizers = constraintNormalizer(id, graph);
         uc.constraintNormalizersPerModel.add(constraintNormalizers);

         System.out.println("Loaded model: " + modelName + " with " + steps + " steps"
               + (conditions != null ? " and conditions" : ""));
      }

      // Set first model as the single inputGraph for compatibility
      uc.inputModel = xmiFiles[0].getPath();
      uc.inputGraph = uc.inputGraphs.get(0);

      return uc;
   }

   /**
    * Calculate the conditions for a given model based on case-specific logic.
    * Returns null if the case does not use conditions.
    * This is a placeholder implementation - logic will be specified later.
    *
    * @param caseId
    *           The case ID (cra, slb, scrum)
    * @param graph
    *           The input graph to calculate conditions for
    * @return JSONObject with conditions for this model, or null if not applicable
    */
   public static double[] objectiveNormalizer(final String caseId, final EGraph graph) {
      final double[] c;
      switch(caseId.toLowerCase()) {
         case "cra":
            // TODO: Specify conditions logic for CRA case
            // Placeholder: no conditions
            return null;

         case "slb":
            // TODO: Specify conditions logic for SLB (Stack) case
            // Placeholder: no conditions
            return null;
         case "scrum":
         case "scrum-uc":
            return null;
         // final Plan p = (Plan) graph.getRoots().get(0);
         // final int numSprints = p.getMaxSprints();
         //
         // final org.apache.commons.math3.stat.descriptive.moment.StandardDeviation stdDevCalculator = new
         // org.apache.commons.math3.stat.descriptive.moment.StandardDeviation();
         //
         // final double worstCaseFitness = p.getStakeholders().stream().mapToDouble(stakeholder -> {
         // // Total importance for this stakeholder across ALL items
         // final double totalImportance = p.getBacklog().getWorkitems().stream()
         // .filter(item -> item.getStakeholder().equals(stakeholder)).mapToInt(item -> item.getImportance())
         // .sum();
         //
         // // Worst case: all importance in one sprint, zero in the rest
         // final double[] worstCaseValues = new double[numSprints];
         // Arrays.fill(worstCaseValues, 0.0);
         // worstCaseValues[0] = totalImportance;
         //
         // return stdDevCalculator.evaluate(worstCaseValues);
         // }).sum();
         //
         // // Total effort across all backlog items
         // final double totalEffort = p.getBacklog().getWorkitems().stream().mapToInt(item -> item.getEffort()).sum();
         //
         // // Worst case: all effort in one sprint, zero in the rest
         // final double[] worstCaseValues = new double[numSprints];
         // Arrays.fill(worstCaseValues, 0.0);
         // worstCaseValues[0] = totalEffort;
         //
         // final double worstCaseEffortDeviation = stdDevCalculator.evaluate(worstCaseValues);
         //
         // c = new double[3];
         // c[0] = worstCaseFitness > 0 ? 1.0 / worstCaseFitness : 1.0;
         // c[1] = worstCaseEffortDeviation > 0 ? 1.0 / worstCaseEffortDeviation : 1.0;
         // c[2] = 1.0; // SolutionLength normalizer

         // return c;
         default:
            // Default: no conditions
            return null;
      }
   }

   public Path testModelPath;

   public Class encodingStrategyClass;
   public ModuleManager moduleManager;

   IEGraphMultiDimensionalFitnessFunction function;
   IEGraphMultiDimensionalFitnessFunction realFunction;

   ToDoubleFunction<double[]> fitnessAggregator;
   ToDoubleFunction<double[]> constraintAggregator;

   ToDoubleFunction<EObject> fitnessScaler;
   ToDoubleFunction<EObject> constraintScaler;

   public String objectiveName;
   public String constraintName;
   public JSONObject conditions;
   public String inputModel;
   public EGraph inputGraph;
   // Multi-model support
   public List<EGraph> inputGraphs;
   public List<String> modelNames;
   public List<Integer> stepsPerModel;
   public List<JSONObject> conditionsPerModel;
   public List<double[]> fitnessNormalizersPerModel;
   public List<double[]> constraintNormalizersPerModel;

   UseCase(final String objectiveName) {
      this(objectiveName, null);
   }

   UseCase(final String objectiveName, final String constraintName) {
      this.objectiveName = objectiveName;
      this.constraintName = constraintName;
   }

}
