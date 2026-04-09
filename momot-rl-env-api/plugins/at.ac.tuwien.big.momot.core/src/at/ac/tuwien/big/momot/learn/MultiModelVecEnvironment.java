package at.ac.tuwien.big.momot.learn;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.SolutionArchive;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.SOEnvironment;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.util.LearnAPIUtils;
import at.ac.tuwien.big.momot.util.MomotUtil;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import javax.annotation.Nullable;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.json.JSONObject;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;
import org.moeaframework.core.comparator.DominanceComparator;

import co.gongzh.procbridge.Delegate;
import co.gongzh.procbridge.Handler;

/**
 * Multi-model variant of NNDelegateVec.
 *
 * Instead of using fixed initial solutions per environment, this version:
 * - Stores a list of input graphs and their pre-calculated step counts
 * - Uses round-robin cycling to assign models to environments during reset
 * - Dynamically updates envMaxSteps based on the current model
 */
public class MultiModelVecEnvironment extends Delegate {
   private final String BASE_PATH = "output/nn_logs";
   private String WRITE_PATH = null;
   private int evalPhases = 0;
   private final String title;
   private AtomicInteger[] noExecutions;
   private final AtomicInteger fe = new AtomicInteger(0);

   private PrintWriter[] fWriter = null;
   private AtomicInteger[] failedCnt;
   ResourceSet[] hrs;

   private final List<SOEnvironment<TransformationSolution>> envs;
   private final List<JSONObject> conditionsPerModel;
   private final int[] envMaxSteps;
   private Population ndp;
   private Population[] ndps;

   protected boolean useArchive = false;
   protected SolutionArchive<TransformationSolution>[] archive = null;
   private final boolean evalMode;
   private final ModelAssignmentPolicy modelAssignmentPolicy;

   private final List<EGraph> inputGraphs;
   private final List<String> modelNames;
   private final List<Integer> stepsPerModel;
   private final List<IEGraphMultiDimensionalFitnessFunction> fitnessFunctions;
   private final List<double[]> fitnessNormalizersPerModel;
   private final List<double[]> constraintNormalizersPerModel;

   private final int[] currentModelIdx; // Track which model each environment is using

   public MultiModelVecEnvironment(final boolean evalMode, final List<SOEnvironment<TransformationSolution>> envs,
         final String title, final List<String> modelNames, final List<EGraph> inputGraphs,
         final List<Integer> stepsPerModel, final List<IEGraphMultiDimensionalFitnessFunction> fitnessFunctions,
         final List<JSONObject> conditionsPerModel, final List<double[]> fitnessNormalizersPerModel,
         final List<double[]> constraintNormalizersPerModel) {
      this.evalMode = evalMode;
      this.envs = envs;
      this.title = title;
      this.inputGraphs = inputGraphs;
      this.stepsPerModel = stepsPerModel;
      this.modelNames = modelNames;
      this.fitnessFunctions = fitnessFunctions;
      this.conditionsPerModel = conditionsPerModel;
      this.fitnessNormalizersPerModel = fitnessNormalizersPerModel;
      this.constraintNormalizersPerModel = constraintNormalizersPerModel;

      this.envMaxSteps = new int[envs.size()];
      this.archive = new SolutionArchive[envs.size()];
      this.currentModelIdx = new int[envs.size()];
      this.modelAssignmentPolicy = new ModelAssignmentPolicy(stepsPerModel, envs.size());

      ndp = new NondominatedPopulation();
      hrs = new ResourceSet[envs.size()];
      fWriter = new PrintWriter[envs.size()];
      noExecutions = new AtomicInteger[envs.size()];
      failedCnt = new AtomicInteger[envs.size()];
      ndps = new Population[envs.size()];

      final Resource.Factory.Registry reg = Resource.Factory.Registry.INSTANCE;
      reg.getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

      IntStream.range(0, envs.size()).forEach(i -> {
         final int modelIdx = evalMode ? i : modelAssignmentPolicy.initModelIndex(i);

         currentModelIdx[i] = modelIdx;

         this.envMaxSteps[i] = stepsPerModel.get(modelIdx);
         failedCnt[i] = new AtomicInteger(0);
         noExecutions[i] = new AtomicInteger(0);
         hrs[i] = new ResourceSetImpl();
         ndps[i] = new NondominatedPopulation();

         final EGraph graph = MomotUtil.copy(inputGraphs.get(modelIdx));
         final IEGraphMultiDimensionalFitnessFunction envFunction = fitnessFunctions.get(i);
         final TransformationSolution initialSolution = MomotUtil.readInitialSolution(graph,
               envFunction.getObjectiveNames().size(), envFunction.getConstraintNames().size());
         envFunction.evaluate(initialSolution);

         envs.get(i).setSolutionLength(envMaxSteps[i]);
         envs.get(i).setInitialSolution(initialSolution);
         if(fitnessNormalizersPerModel != null && !fitnessNormalizersPerModel.isEmpty()) {
            envs.get(i).setFitnessNormalizers(fitnessNormalizersPerModel.get(modelIdx));
         }
         if(constraintNormalizersPerModel != null && !constraintNormalizersPerModel.isEmpty()) {
            envs.get(i).setConstraintNormalizers(constraintNormalizersPerModel.get(modelIdx));
         }

         if(useArchive) {
            this.archive[i] = new SolutionArchive<>();
         }

         System.out.println(
               "Environment " + i + " initialized with model " + modelIdx + " (steps: " + envMaxSteps[i] + ")");
      });
   }

   protected void addSolutionIfValid(final int envIdx, final Solution s) {
      if(!this.hasNaNOrInfinity(s.getObjectives())) {
         this.ndps[envIdx].add(s);
      }
   }

   private double clipReward(final double reward, final double min, final double max) {
      return Math.max(min, Math.min(max, reward));
   }

   @Handler
   Object echo(final Object payload) {
      return payload;
   }

   @Handler
   void err() {
      throw new RuntimeException("generated error");
   }

   @Handler
   String getEnvModelName(final int envIdx) {
      return this.modelNames.get(envIdx);
   }

   @Handler
   List<double[]> getEnvSolutionsAndClear(final int envIdx) {
      final List<double[]> solutions = new ArrayList<>();
      for(final Solution solution : this.ndps[envIdx]) {
         solutions.add(solution.getObjectives());
      }
      this.ndps[envIdx].clear();

      return solutions;
   }

   @Handler
   int getMaxSteps() {
      return this.envMaxSteps[0];
   }

   @Override
   protected @Nullable Object handleUnknownRequest(@Nullable final String method, @Nullable final Object payload) {
      return null;
   }

   private boolean hasNaNOrInfinity(final double[] arr) {
      for(final double v : arr) {
         if(v != v || v == Double.POSITIVE_INFINITY || v == Double.NEGATIVE_INFINITY || Double.isNaN(v)) {
            return true;
         }
      }
      return false;
   }

   protected boolean isDominatedByAnySolutionInParetoFront(final Solution s) {

      final DominanceComparator comparator = ((NondominatedPopulation) this.ndp).getComparator();
      for(int i = 0; i < this.ndp.size(); i++) {
         if(comparator.compare(s, this.ndp.get(i)) > 0) {
            return true;
         }
      }
      return false;
   }

   @Handler
   int numInstances() {
      return this.inputGraphs.size();
   }

   @Handler
   void processEnvSolutions(final boolean clear) {
      final Population aggPop = new NondominatedPopulation();
      for(final Population ndp2 : this.ndps) {
         aggPop.addAll(ndp2);
         if(clear) {
            ndp2.clear();
         }
      }
      writeSolutions(aggPop, "running-solutions");

      // IntStream.range(0, envs.size()).forEach(i -> {
      // envs.get(i).adaptLagragian();
      // });
   }

   @Handler
   List<double[]> processSolutions(final boolean finished) {
      for(int i = 0; i < this.ndps.length; i++) {
         this.ndp.addAll(this.ndps[i]);
         writeEnvSolutions(i, this.ndps[i]);
         this.ndps[i].clear();
      }

      writeSolutions(this.ndp, finished ? "nn-solutions_final" : "nn-solutions");

      final List<double[]> fSolutions = new ArrayList<>();
      for(int i = 0; i < this.ndp.size(); i++) {
         fSolutions.add(this.ndp.get(i).getObjectives());
      }

      if(finished) {
         this.ndp.clear();
      }

      if(evalMode) {
         evalPhases++;
      }

      return fSolutions;
   }

   @Handler
   Object[] reset(final int envIdx) {
      final Object[] rv = new Object[2];

      try {
         failedCnt[envIdx].set(0);
         noExecutions[envIdx].set(0);

         final int modelIdx = evalMode ? envIdx : modelAssignmentPolicy.nextModelIndex(envIdx);
         currentModelIdx[envIdx] = modelIdx;

         // Update max steps for this environment based on the model
         this.envMaxSteps[envIdx] = stepsPerModel.get(modelIdx);
         envs.get(envIdx).setSolutionLength(envMaxSteps[envIdx]);
         if(fitnessNormalizersPerModel != null && !fitnessNormalizersPerModel.isEmpty()) {
            envs.get(envIdx).setFitnessNormalizers(fitnessNormalizersPerModel.get(modelIdx));
         }
         if(constraintNormalizersPerModel != null && !constraintNormalizersPerModel.isEmpty()) {
            envs.get(envIdx).setConstraintNormalizers(constraintNormalizersPerModel.get(modelIdx));
         }

         System.out.println("------------------------------------ RESET ENV " + envIdx + " WITH MODEL " + modelIdx
               + " (steps: " + envMaxSteps[envIdx] + ") ------------------------------------");

         // Create new initial solution from the model
         final EGraph graph = MomotUtil.copy(inputGraphs.get(modelIdx));
         final IEGraphMultiDimensionalFitnessFunction envFunction = fitnessFunctions.get(envIdx);
         final TransformationSolution initialSolution = MomotUtil.readInitialSolution(graph,
               envFunction.getObjectiveNames().size(), envFunction.getConstraintNames().size());
         envFunction.evaluate(initialSolution);

         if(useArchive && archive[envIdx] != null && archive[envIdx].size() > 0) {
            final TransformationSolution resetToState = this.archive[envIdx].retrieve(envs.get(envIdx).getEngine());
            envs.get(envIdx).setInitialSolution(resetToState);
            noExecutions[envIdx].set(resetToState.getNumberOfVariables());
            System.out.println(
                  "Reset " + envIdx + " to archived solution of length " + resetToState.getNumberOfVariables());
         } else {
            envs.get(envIdx).setInitialSolution(initialSolution);
         }

         final TransformationSolution ts = envs.get(envIdx).reset();
         System.out.println("Solution @ " + envIdx + " has length " + ts.getSolutionLength());

         final Resource res = hrs[envIdx].createResource(URI.createFileURI("http://MyModel" + envIdx + ".xmi"));
         final List<EObject> encodeGraph = ts.getResultGraph() != null ? ts.getResultGraph().getRoots()
               : ts.getSourceGraph().getRoots();

         res.getContents().addAll(encodeGraph);
         LearnAPIUtils.writeFileAndConsole(fWriter[envIdx],
               envs.get(envIdx).getProblemEncoder().printModel(envs.get(envIdx).getInitialSolution()));

         rv[0] = LearnAPIUtils.transformXMIToString(res);

         if(this.conditionsPerModel != null && modelIdx < this.conditionsPerModel.size()
               && this.conditionsPerModel.get(modelIdx) != null) {
            rv[1] = this.conditionsPerModel.get(modelIdx);
         }
      } catch(final Exception e) {
         e.printStackTrace();
      }

      return rv;
   }

   @Handler
   void resetInstance(final String logPath) {
      WRITE_PATH = Path.of(BASE_PATH, title, logPath).toString();
      fe.set(0);
      modelAssignmentPolicy.reset();

      ndp = new NondominatedPopulation();
      hrs = new ResourceSet[envs.size()];
      fWriter = new PrintWriter[envs.size()];
      noExecutions = new AtomicInteger[envs.size()];
      failedCnt = new AtomicInteger[envs.size()];
      ndps = new Population[envs.size()];

      IntStream.range(0, envs.size()).forEach(i -> {
         failedCnt[i] = new AtomicInteger(0);
         noExecutions[i] = new AtomicInteger(0);
         hrs[i] = new ResourceSetImpl();
         ndps[i] = new NondominatedPopulation();
      });

      final Resource.Factory.Registry reg = Resource.Factory.Registry.INSTANCE;
      reg.getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

      try {
         Files.createDirectories(Paths.get(WRITE_PATH));

         IntStream.range(0, envs.size()).forEach(i -> {
            try {
               fWriter[i] = new PrintWriter(new FileWriter(WRITE_PATH + "/nn" + i + ".txt"));
            } catch(final IOException e) {
               e.printStackTrace();
            }
         });
      } catch(final IOException e) {
         e.printStackTrace();
      }
   }

   private int selectModelIndexByInverseHorizon() {
      final int size = stepsPerModel.size();
      if(size == 0) {
         return 0;
      }

      double totalWeight = 0.0;
      final double[] weights = new double[size];
      for(int i = 0; i < size; i++) {
         final int steps = stepsPerModel.get(i);
         final double weight = steps > 0 ? 1.0 / steps : 0.0;
         weights[i] = weight;
         totalWeight += weight;
      }

      if(totalWeight <= 0.0) {
         return ThreadLocalRandom.current().nextInt(size);
      }

      double r = ThreadLocalRandom.current().nextDouble(totalWeight);
      for(int i = 0; i < size; i++) {
         r -= weights[i];
         if(r <= 0.0) {
            return i;
         }
      }

      return size - 1;
   }

   @Handler
   Object[] step(final JSONObject payload) {
      final int envIdx = payload.getInt("_envId");

      try {
         final String rule = payload.getString("_ruleName");
         final HashMap<String, Object> paramValues = new ObjectMapper()
               .readValue(payload.getJSONObject("_paramValues").toString(), HashMap.class);

         LearnAPIUtils.writeFileAndConsole(fWriter[envIdx],
               "\nR: " + rule.substring(rule.lastIndexOf("::") + 2) + " (env " + envIdx + ")");
         paramValues.forEach((key, value) -> LearnAPIUtils.writeFileAndConsole(fWriter[envIdx],
               key + " " + value + " (env " + envIdx + ")"));

         final SOEnvResponse<TransformationSolution> r = step(envs.get(envIdx), rule, paramValues);
         final Object[] rv = new Object[3];

         rv[1] = r.getReward();
         noExecutions[envIdx].incrementAndGet();

         if(r.getState() == null) {
            rv[0] = false;
            rv[1] = r.getReward() - 5;
            LearnAPIUtils.writeFileAndConsole(fWriter[envIdx], "Failed");
         } else {
            fe.incrementAndGet();
            failedCnt[envIdx].set(0);

            addSolutionIfValid(envIdx, r.getState());

            final Resource res = hrs[envIdx].createResource(URI.createFileURI("http://MyModel" + envIdx + ".xmi"));
            res.getContents().addAll(r.getState().getResultGraph().getRoots());

            rv[0] = LearnAPIUtils.transformXMIToString(res);

            res.getContents().clear();
            res.unload();
            hrs[envIdx].getResources().clear();

            final double[] fitnessVals = envs.get(envIdx).retrieveObjectives(r.getState());
            final double aggrFitness = -Arrays.stream(fitnessVals).sum();
            if(useArchive && !this.hasNaNOrInfinity(fitnessVals)) {
               archive[envIdx].addOrUpdate(r.getState(), r.getState().getNumberOfVariables(), aggrFitness);
            }

            LearnAPIUtils.writeFileAndConsole(fWriter[envIdx],
                  envs.get(envIdx).getProblemEncoder().printModel(r.getState()));
            LearnAPIUtils.writeFileAndConsole(fWriter[envIdx],
                  "len: " + r.getState().getSolutionLength() + " (R " + rv[1] + ")");
         }

         if(noExecutions[envIdx].get() >= this.envMaxSteps[envIdx]) {
            System.out.println("Reset @ steps = " + noExecutions[envIdx].get());
            rv[2] = true;
            envs.get(envIdx).endEpisode();
         } else {
            System.out.println("current steps = " + noExecutions[envIdx].get());
            rv[2] = false;
         }

         return rv;

      } catch(final Exception e) {
         LearnAPIUtils.writeFileAndConsole(fWriter[envIdx], "ERR:");
         LearnAPIUtils.writeFileAndConsole(fWriter[envIdx], e.getMessage());
         e.printStackTrace();
         throw new RuntimeException(e);
      }
   }

   protected SOEnvResponse<TransformationSolution> step(final SOEnvironment<TransformationSolution> env,
         final String rule, final HashMap<String, Object> paramValues) {
      return env.step(rule, paramValues);
   }

   private synchronized void writeEnvSolutions(final int envIdx, final Population p) {
      final StringBuilder sb = new StringBuilder("\n");
      final String header = evalMode ? String.format("%d. evaluation\t(%d fes)\n", evalPhases + 1, fe.get())
            : String.format("%d\n", fe.get());
      sb.append(header);
      for(final Solution s : p) {
         sb.append(Arrays.toString(s.getObjectives()) + "\n");
      }
      final Path writeBasePath = Path.of(WRITE_PATH, Integer.toString(envIdx) + "-nn-solutions_final");

      try(PrintWriter out = new PrintWriter(new FileWriter(writeBasePath.toString() + ".txt", true))) {
         out.println(sb.toString());
      } catch(final IOException e) {
         e.printStackTrace();
      }
   }

   private synchronized void writeSolutions(final Population p, final String fileName) {
      final StringBuilder sb = new StringBuilder("\n" + fe + "\n");
      for(final Solution s : p) {
         sb.append(Arrays.toString(s.getObjectives()) + "\n");
      }
      final Path writeBasePath = Path.of(WRITE_PATH, fileName);

      try(PrintWriter out = new PrintWriter(new FileWriter(writeBasePath.toString() + ".txt", true))) {
         out.println(sb.toString());
      } catch(final IOException e) {
         e.printStackTrace();
      }
   }

}
