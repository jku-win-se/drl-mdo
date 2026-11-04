package at.ac.tuwien.big.momot.learn;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.SOEnvironment;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.util.LearnAPIUtils;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.BufferedWriter;
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
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import javax.annotation.Nullable;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.json.JSONArray;
import org.json.JSONObject;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;

import co.gongzh.procbridge.Delegate;
import co.gongzh.procbridge.Handler;

/**
 * Multi-model variant of NNDelegateVec.
 *
 * Stores a list of input graphs and their pre-calculated step counts.
 * Assigns models to environments with DETERMINISTIC_PER_ENV_AND_INSTANCE
 * (envIdx mod number of models), or moves training envs through all models with CYCLE,
 * and updates envMaxSteps based on the current model.
 */
public class MultiModelVecEnvironment extends Delegate {
   private static final boolean STEP_LOG = Boolean.getBoolean("momot.stepLog");
   private static final int DEFAULT_MAX_FAILED_RETRIES = 20;

   private final String BASE_PATH = "output/nn_logs";
   private String WRITE_PATH = null;
   private int evalPhases = 0;
   private final String title;
   private AtomicInteger[] noExecutions;
   private int[] failedInARow;
   private int maxFailedRetries;
   private final AtomicInteger fe = new AtomicInteger(0);

   private PrintWriter[] fWriter = null;
   ResourceSet[] hrs;

   private final List<SOEnvironment<TransformationSolution>> envs;
   private final List<JSONObject> conditionsPerModel;
   private final int[] envMaxSteps;
   private Population ndp;
   private Population[] ndps;

   private final boolean evalMode;
   private final ModelAssignmentPolicy modelAssignmentPolicy;

   private final List<EGraph> inputGraphs;
   private final List<String> modelNames;
   private final List<Integer> stepsPerModel;
   private final List<IEGraphMultiDimensionalFitnessFunction> fitnessFunctions;
   private final List<double[]> fitnessNormalizersPerModel;
   private final List<double[]> constraintNormalizersPerModel;

   private final int[] currentModelIdx;
   private final Object[] envLocks;
   private final ExecutorService stepPool;
   private volatile CompactGraphEncoder graphEncoder;
   // Set by the client through configureObservation: an action that cannot be applied uses one step.
   private volatile boolean invalidActionCostsStep = false;
   private JSONObject runInfo = new JSONObject();

   public MultiModelVecEnvironment(final boolean evalMode, final List<SOEnvironment<TransformationSolution>> envs,
         final String title, final List<String> modelNames, final List<EGraph> inputGraphs,
         final List<Integer> stepsPerModel, final List<IEGraphMultiDimensionalFitnessFunction> fitnessFunctions,
         final List<JSONObject> conditionsPerModel, final List<double[]> fitnessNormalizersPerModel,
         final List<double[]> constraintNormalizersPerModel) {
      this(evalMode, envs, title, modelNames, inputGraphs, stepsPerModel, fitnessFunctions, conditionsPerModel,
            fitnessNormalizersPerModel, constraintNormalizersPerModel, false);
   }

   /**
    * @param cycleModels
    *           Training envs move through the models instead of each staying on one.
    */
   public MultiModelVecEnvironment(final boolean evalMode, final List<SOEnvironment<TransformationSolution>> envs,
         final String title, final List<String> modelNames, final List<EGraph> inputGraphs,
         final List<Integer> stepsPerModel, final List<IEGraphMultiDimensionalFitnessFunction> fitnessFunctions,
         final List<JSONObject> conditionsPerModel, final List<double[]> fitnessNormalizersPerModel,
         final List<double[]> constraintNormalizersPerModel, final boolean cycleModels) {
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

      // Evaluation envs are pointed at one model at a time, so any env count fits.
      if(!evalMode && !cycleModels && stepsPerModel != null && !stepsPerModel.isEmpty()
            && envs.size() % stepsPerModel.size() != 0) {
         throw new IllegalArgumentException("n_envs=" + envs.size()
               + " must be a multiple of the number of models=" + stepsPerModel.size());
      }

      this.envMaxSteps = new int[envs.size()];
      this.currentModelIdx = new int[envs.size()];
      this.envLocks = new Object[envs.size()];
      for(int i = 0; i < envs.size(); i++) {
         this.envLocks[i] = new Object();
      }
      this.modelAssignmentPolicy = new ModelAssignmentPolicy(stepsPerModel, envs.size());
      if(cycleModels) {
         this.modelAssignmentPolicy.setStrategy(ModelAssignmentPolicy.Strategy.CYCLE);
      }
      this.stepPool = Executors.newFixedThreadPool(Integer.getInteger("momot.stepThreads", envs.size()),
            daemonThreadFactory("momot-step"));
      this.maxFailedRetries = DEFAULT_MAX_FAILED_RETRIES;
      this.failedInARow = new int[envs.size()];

      ndp = new NondominatedPopulation();
      hrs = new ResourceSet[envs.size()];
      fWriter = new PrintWriter[envs.size()];
      noExecutions = new AtomicInteger[envs.size()];
      ndps = new Population[envs.size()];

      final Resource.Factory.Registry reg = Resource.Factory.Registry.INSTANCE;
      reg.getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

      IntStream.range(0, envs.size()).forEach(i -> {
         final int modelIdx = modelAssignmentPolicy.initModelIndex(i);

         currentModelIdx[i] = modelIdx;

         this.envMaxSteps[i] = stepsPerModel.get(modelIdx);
         noExecutions[i] = new AtomicInteger(0);
         hrs[i] = new ResourceSetImpl();
         ndps[i] = new NondominatedPopulation();

         final EGraph graph = inputGraphs.get(modelIdx);
         final IEGraphMultiDimensionalFitnessFunction envFunction = fitnessFunctions.get(i);
         final TransformationSolution initialSolution = MomotUtil.readInitialSolution(graph,
               envFunction.getObjectiveNames().size(), envFunction.getConstraintNames().size());
         envFunction.evaluate(initialSolution);

         envs.get(i).setSolutionLength(envMaxSteps[i]);
         envs.get(i).setInitialSolution(initialSolution);
         envs.get(i).setApplyRulesInPlace(true);
         if(fitnessNormalizersPerModel != null && !fitnessNormalizersPerModel.isEmpty()) {
            envs.get(i).setFitnessNormalizers(fitnessNormalizersPerModel.get(modelIdx));
         }
         if(constraintNormalizersPerModel != null && !constraintNormalizersPerModel.isEmpty()) {
            envs.get(i).setConstraintNormalizers(constraintNormalizersPerModel.get(modelIdx));
         }

         System.out.println(
               "Environment " + i + " initialized with model " + modelIdx + " (steps: " + envMaxSteps[i] + ")");
      });
   }

   public MultiModelVecEnvironment(final boolean evalMode, final List<SOEnvironment<TransformationSolution>> envs,
         final String title, final List<String> modelNames, final List<EGraph> inputGraphs,
         final List<Integer> stepsPerModel, final List<IEGraphMultiDimensionalFitnessFunction> fitnessFunctions,
         final List<JSONObject> conditionsPerModel, final List<double[]> fitnessNormalizersPerModel,
         final List<double[]> constraintNormalizersPerModel, final int maxFailedRetries) {
      this(evalMode, envs, title, modelNames, inputGraphs, stepsPerModel, fitnessFunctions, conditionsPerModel,
            fitnessNormalizersPerModel, constraintNormalizersPerModel);
      if(maxFailedRetries < 1) {
         throw new IllegalArgumentException(
               "maxFailedRetries must be at least 1, got " + maxFailedRetries);
      }
      this.maxFailedRetries = maxFailedRetries;
   }

   /** Only feasible states that changed the model can be results. */
   protected void addSolutionIfValid(final int envIdx, final TransformationSolution s) {
      if(!this.hasNaNOrInfinity(s.getObjectives()) && !s.violatesConstraints() && s.getSolutionLength() > 0) {
         this.ndps[envIdx].add(new Solution(s.getObjectives()));
      }
   }

   public void close() {
      flushWriters();
      closeWriters();
      if(stepPool != null) {
         stepPool.shutdownNow();
         try {
            stepPool.awaitTermination(2, TimeUnit.SECONDS);
         } catch(final InterruptedException e) {
            Thread.currentThread().interrupt();
         }
      }
   }

   @Handler
   void configureObservation(final JSONObject spec) {
      this.graphEncoder = CompactGraphEncoder.fromSpec(spec);
      this.invalidActionCostsStep = spec.optBoolean("invalidActionCostsStep", false);
   }

   private static ThreadFactory daemonThreadFactory(final String name) {
      return r -> {
         final Thread t = new Thread(r, name);
         t.setDaemon(true);
         return t;
      };
   }

   private void closeWriters() {
      if(fWriter == null) {
         return;
      }
      for(int i = 0; i < fWriter.length; i++) {
         if(fWriter[i] != null) {
            fWriter[i].close();
            fWriter[i] = null;
         }
      }
   }

   @Handler
   Object echo(final Object payload) {
      return payload;
   }

   @Handler
   void err() {
      throw new RuntimeException("generated error");
   }

   private void flushWriters() {
      if(fWriter == null) {
         return;
      }
      for(final PrintWriter writer : fWriter) {
         if(writer != null) {
            writer.flush();
         }
      }
   }

   @Handler
   String getEnvModelName(final int envIdx) {
      return this.modelNames.get(currentModelIdx[envIdx]);
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

   /** Set once by the server's main: what the client would otherwise have to be told separately. */
   public void setRunInfo(final JSONObject runInfo) {
      this.runInfo = runInfo;
   }

   /** The case, the port of each evaluation dataset, and the dataset paths. */
   @Handler
   JSONObject getRunInfo() {
      return this.runInfo;
   }

   /** Steps that returned a successor model since resetInstance. */
   @Handler
   int getFeCount() {
      return this.fe.get();
   }

   @Handler
   int numEnvs() {
      return this.envs.size();
   }

   /** Every model of this dataset with its horizon, whether or not an env currently plays it. */
   @Handler
   JSONObject getModelNames() {
      final JSONObject out = new JSONObject();
      out.put("names", new JSONArray(this.modelNames));
      out.put("horizons", new JSONArray(this.stepsPerModel));
      return out;
   }

   /**
    * Returns the largest step budget over all models of this dataset.
    * Individual environments may have a smaller budget depending on their model.
    */
   @Handler
   int getMaxSteps() {
      int max = 0;
      for(final int steps : this.stepsPerModel) {
         if(steps > max) {
            max = steps;
         }
      }
      return max;
   }

   /**
    * Per-environment step budgets. Returns a copy so callers cannot mutate the live counters.
    * Older jars have no such handler and answer the request with null.
    */
   @Handler
   List<Integer> getEnvMaxSteps() {
      final List<Integer> steps = new ArrayList<>(this.envMaxSteps.length);
      for(final int stepsForEnv : this.envMaxSteps) {
         steps.add(stepsForEnv);
      }
      return steps;
   }

   @Override
   protected @Nullable Object handleUnknownRequest(@Nullable final String method, @Nullable final Object payload) {
      return null;
   }

   private boolean hasNaNOrInfinity(final double[] arr) {
      for(final double v : arr) {
         if(Double.isNaN(v) || Double.isInfinite(v)) {
            return true;
         }
      }
      return false;
   }

   private void log(final int envIdx, final String s) {
      if(!STEP_LOG || fWriter == null || fWriter[envIdx] == null || s == null) {
         return;
      }
      synchronized(fWriter[envIdx]) {
         fWriter[envIdx].println(s);
      }
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
      flushWriters();
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

      flushWriters();
      return fSolutions;
   }

   @Handler
   Object[] reset(final int envIdx) {
      return resetEnv(envIdx, false);
   }

   @Handler
   JSONArray resetAll() {
      ensureObservationEncoderConfigured();
      final List<Callable<Object[]>> tasks = new ArrayList<>(envs.size());
      for(int i = 0; i < envs.size(); i++) {
         final int envIdx = i;
         tasks.add(() -> resetEnv(envIdx, true));
      }
      return asJsonArray(invokeAll(tasks));
   }

   /** Point every env at one model and reset it, so all of them play that model next. */
   @Handler
   JSONArray resetEnvsToModel(final int modelIdx) {
      ensureObservationEncoderConfigured();
      final List<Callable<Object[]>> tasks = new ArrayList<>(envs.size());
      for(int i = 0; i < envs.size(); i++) {
         final int envIdx = i;
         tasks.add(() -> resetEnv(envIdx, true, modelIdx));
      }
      return asJsonArray(invokeAll(tasks));
   }

   @Handler
   void resetInstance(final String logPath) {
      WRITE_PATH = Path.of(BASE_PATH, title, logPath).toString();
      fe.set(0);
      modelAssignmentPolicy.reset();

      ndp = new NondominatedPopulation();
      closeWriters();
      hrs = new ResourceSet[envs.size()];
      fWriter = new PrintWriter[envs.size()];
      noExecutions = new AtomicInteger[envs.size()];
      ndps = new Population[envs.size()];

      IntStream.range(0, envs.size()).forEach(i -> {
         noExecutions[i] = new AtomicInteger(0);
         failedInARow[i] = 0;
         hrs[i] = new ResourceSetImpl();
         ndps[i] = new NondominatedPopulation();
         currentModelIdx[i] = modelAssignmentPolicy.initModelIndex(i);
      });

      try {
         Files.createDirectories(Paths.get(WRITE_PATH));

         if(STEP_LOG) {
            IntStream.range(0, envs.size()).forEach(i -> {
               try {
                  fWriter[i] = new PrintWriter(new BufferedWriter(new FileWriter(WRITE_PATH + "/nn" + i + ".txt")));
               } catch(final IOException e) {
                  e.printStackTrace();
               }
            });
         }
      } catch(final IOException e) {
         e.printStackTrace();
      }
   }

   @Handler
   Object[] step(final JSONObject payload) {
      final Object[] full = stepEnv(payload.getInt("_envId"), payload.getString("_ruleName"),
            jsonObjectToMap(payload.getJSONObject("_paramValues")), false, false);
      return new Object[] { full[0], full[1], full[2] };
   }

   @Handler
   JSONArray stepAll(final JSONObject payload) {
      ensureObservationEncoderConfigured();
      final JSONArray actions = payload.getJSONArray("_actions");
      final boolean autoReset = payload.optBoolean("_autoReset", true);
      final List<Callable<Object[]>> tasks = new ArrayList<>(actions.length());
      for(int i = 0; i < actions.length(); i++) {
         final JSONObject action = actions.getJSONObject(i);
         tasks.add(() -> {
            final int envIdx = action.getInt("_envId");
            if(action.optBoolean("_skip", false)) {
               return skipEnv(envIdx, autoReset);
            }
            return stepEnv(envIdx, action.getString("_ruleName"),
                  jsonObjectToMap(action.getJSONObject("_paramValues")), true, autoReset);
         });
      }
      return asJsonArray(invokeAll(tasks));
   }

   /** The client found no legal action for this env, so nothing reached the environment. */
   private Object[] skipEnv(final int envIdx, final boolean autoReset) {
      synchronized(envLocks[envIdx]) {
         final JSONObject info = new JSONObject();
         info.put("model", currentModelIdx[envIdx]);
         info.put("invalid", true);
         if(!invalidActionCostsStep) {
            return new Object[] { JSONObject.NULL, 0.0, false, JSONObject.NULL, info };
         }
         return invalidStep(envIdx, true, autoReset, info);
      }
   }

   /** One used step that leaves the model unchanged, returned as an ordinary transition. */
   private Object[] invalidStep(final int envIdx, final boolean compact, final boolean autoReset,
         final JSONObject info) {
      failedInARow[envIdx] = 0;
      noExecutions[envIdx].incrementAndGet();
      final Object[] rv = new Object[5];
      rv[0] = encodeObservation(envIdx, envs.get(envIdx).getCurrentState(), compact);
      rv[1] = 0.0;
      rv[4] = info;
      return finishStep(envIdx, compact, autoReset, rv);
   }

   private Object[] finishStep(final int envIdx, final boolean compact, final boolean autoReset, final Object[] rv) {
      final boolean done = noExecutions[envIdx].get() >= this.envMaxSteps[envIdx];
      rv[2] = done;
      if(done && STEP_LOG) {
         flushWriters();
      }

      if(autoReset && done) {
         final Object[] resetRv = resetEnv(envIdx, compact);
         rv[0] = resetRv[0];
         rv[3] = resetRv[1];
         final JSONObject info = (JSONObject) rv[4];
         info.put("next_model", currentModelIdx[envIdx]);
         info.put("next_horizon", envMaxSteps[envIdx]);
      }

      return rv;
   }

   protected SOEnvResponse<TransformationSolution> step(final SOEnvironment<TransformationSolution> env,
         final String rule, final Map<String, Object> paramValues) {
      return env.step(rule, paramValues);
   }

   private Object[] resetEnv(final int envIdx, final boolean compact) {
      synchronized(envLocks[envIdx]) {
         // A reset without any step played does not move the env on to its next model.
         final boolean played = noExecutions[envIdx].get() > 0;
         return resetEnv(envIdx, compact, played ? modelAssignmentPolicy.nextModelIndex(envIdx) : currentModelIdx[envIdx]);
      }
   }

   private Object[] resetEnv(final int envIdx, final boolean compact, final int modelIdx) {
      synchronized(envLocks[envIdx]) {
         final Object[] rv = new Object[3];
         try {
            noExecutions[envIdx].set(0);
            failedInARow[envIdx] = 0;

            currentModelIdx[envIdx] = modelIdx;

            this.envMaxSteps[envIdx] = stepsPerModel.get(modelIdx);
            envs.get(envIdx).setSolutionLength(envMaxSteps[envIdx]);
            if(fitnessNormalizersPerModel != null && !fitnessNormalizersPerModel.isEmpty()) {
               envs.get(envIdx).setFitnessNormalizers(fitnessNormalizersPerModel.get(modelIdx));
            }
            if(constraintNormalizersPerModel != null && !constraintNormalizersPerModel.isEmpty()) {
               envs.get(envIdx).setConstraintNormalizers(constraintNormalizersPerModel.get(modelIdx));
            }

            log(envIdx, "RESET ENV " + envIdx + " WITH MODEL " + modelIdx + " (steps: " + envMaxSteps[envIdx] + ")");

            final EGraph graph = inputGraphs.get(modelIdx);
            final IEGraphMultiDimensionalFitnessFunction envFunction = fitnessFunctions.get(envIdx);
            final TransformationSolution initialSolution = MomotUtil.readInitialSolution(graph,
                  envFunction.getObjectiveNames().size(), envFunction.getConstraintNames().size());
            envFunction.evaluate(initialSolution);

            envs.get(envIdx).setInitialSolution(initialSolution);

            final TransformationSolution ts = envs.get(envIdx).reset();
            log(envIdx, "Solution @ " + envIdx + " has length " + ts.getSolutionLength());
            log(envIdx, envs.get(envIdx).getProblemEncoder().printModel(envs.get(envIdx).getInitialSolution()));

            rv[0] = encodeObservation(envIdx, ts, compact);

            if(this.conditionsPerModel != null && modelIdx < this.conditionsPerModel.size()
                  && this.conditionsPerModel.get(modelIdx) != null) {
               rv[1] = this.conditionsPerModel.get(modelIdx);
            }

            final JSONObject info = new JSONObject();
            info.put("model", modelIdx);
            info.put("horizon", envMaxSteps[envIdx]);
            rv[2] = info;
         } catch(final Exception e) {
            e.printStackTrace();
         }
         return rv;
      }
   }

   private Object[] stepEnv(final int envIdx, final String rule, final Map<String, Object> paramValues,
         final boolean compact, final boolean autoReset) {
      synchronized(envLocks[envIdx]) {
         try {
            log(envIdx, "\nR: " + rule.substring(rule.lastIndexOf("::") + 2) + " (env " + envIdx + ")");
            paramValues.forEach((key, value) -> log(envIdx, key + " " + value + " (env " + envIdx + ")"));

            final JSONObject info = new JSONObject();
            info.put("model", currentModelIdx[envIdx]);

            final SOEnvResponse<TransformationSolution> r = step(envs.get(envIdx), rule, paramValues);
            final Object[] rv = new Object[5];
            rv[4] = info;

            // A rule Henshin cannot apply is not a transition: no penalty and no budget use.
            // After maxFailedRetries failures in a row, count one step so the env cannot stall.
            if(r.getState() == null) {
               log(envIdx, "Failed");
               info.put("invalid", true);
               if(invalidActionCostsStep) {
                  return invalidStep(envIdx, compact, autoReset, info);
               }
               failedInARow[envIdx] = failedInARow[envIdx] + 1;
               if(failedInARow[envIdx] < this.maxFailedRetries) {
                  return new Object[] { false, 0.0, false, null, info };
               }
               failedInARow[envIdx] = 0;
               noExecutions[envIdx].incrementAndGet();
               System.out.println("Warning: environment " + envIdx + " failed " + this.maxFailedRetries
                     + " transformations in a row; counting this attempt as one step with reward 0.");
               rv[0] = false;
               rv[1] = 0.0;
            } else {
               failedInARow[envIdx] = 0;
               noExecutions[envIdx].incrementAndGet();
               fe.incrementAndGet();
               addSolutionIfValid(envIdx, r.getState());
               rv[0] = encodeObservation(envIdx, r.getState(), compact);
               rv[1] = r.getReward();
               log(envIdx, envs.get(envIdx).getProblemEncoder().printModel(r.getState()));
               log(envIdx, "len: " + r.getState().getSolutionLength() + " (R " + rv[1] + ")");
            }

            return finishStep(envIdx, compact, autoReset, rv);
         } catch(final Exception e) {
            log(envIdx, "ERR:");
            log(envIdx, e.getMessage());
            e.printStackTrace();
            throw new RuntimeException(e);
         }
      }
   }

   private Object encodeObservation(final int envIdx, final TransformationSolution ts, final boolean compact) {
      final EGraph graph = ts.getResultGraph() != null ? ts.getResultGraph() : ts.getSourceGraph();
      if(compact) {
         ensureObservationEncoderConfigured();
         return graphEncoder.encode(graph);
      }
      final Resource res = hrs[envIdx].createResource(URI.createFileURI("http://MyModel" + envIdx + ".xmi"));
      res.getContents().addAll(graph.getRoots());
      final String xmi = LearnAPIUtils.transformXMIToString(res);
      res.getContents().clear();
      res.unload();
      hrs[envIdx].getResources().clear();
      return xmi;
   }

   private void ensureObservationEncoderConfigured() {
      if(graphEncoder == null) {
         throw new IllegalStateException("configureObservation must be called before compact observations");
      }
   }

   private static Map<String, Object> jsonObjectToMap(final JSONObject params) {
      final Map<String, Object> paramValues = new HashMap<>();
      if(params == null) {
         return paramValues;
      }
      for(final String key : params.keySet()) {
         paramValues.put(key, params.get(key));
      }
      return paramValues;
   }

   private <T> List<T> invokeAll(final List<Callable<T>> tasks) {
      try {
         final List<Future<T>> futures = stepPool.invokeAll(tasks);
         final List<T> results = new ArrayList<>(futures.size());
         for(final Future<T> f : futures) {
            results.add(f.get());
         }
         return results;
      } catch(final InterruptedException e) {
         Thread.currentThread().interrupt();
         throw new RuntimeException(e);
      } catch(final ExecutionException e) {
         final Throwable cause = e.getCause() != null ? e.getCause() : e;
         if(cause instanceof RuntimeException) {
            throw (RuntimeException) cause;
         }
         throw new RuntimeException(cause);
      }
   }

   private static JSONArray asJsonArray(final List<Object[]> rows) {
      final JSONArray out = new JSONArray();
      for(final Object[] row : rows) {
         final JSONArray a = new JSONArray();
         if(row != null) {
            for(final Object elem : row) {
               a.put(elem == null ? JSONObject.NULL : elem);
            }
         }
         out.put(a);
      }
      return out;
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
