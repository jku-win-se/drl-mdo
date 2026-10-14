package at.ac.tuwien.big.momot.learn;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.SolutionArchive;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.SOEnvironment;
import at.ac.tuwien.big.momot.util.LearnAPIUtils;

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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import javax.annotation.Nullable;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.json.JSONObject;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;
import org.moeaframework.core.comparator.DominanceComparator;

import co.gongzh.procbridge.Delegate;
import co.gongzh.procbridge.Handler;

public class SingleModelVecEnvironment extends Delegate {
   private static final int DEFAULT_MAX_FAILED_RETRIES = 20;
   private static String BASE_PATH = "output/nn_logs";
   private static String WRITE_PATH = null;
   private final String title;
   private AtomicInteger[] noExecutions;
   private int[] failedInARow;
   private int maxFailedRetries;

   private final AtomicInteger fe = new AtomicInteger(0);

   private PrintWriter[] fWriter = null;

   private AtomicInteger[] failedCnt;

   ResourceSet[] hrs;

   private final List<SOEnvironment<TransformationSolution>> envs;
   private final List<JSONObject> conditionList;
   private final int[] envMaxSteps;
   private final TransformationSolution[] initialSolutions;
   private Population ndp;
   private Population[] ndps;

   protected boolean useArchive = false;
   protected SolutionArchive<TransformationSolution>[] archive = null;

   public SingleModelVecEnvironment(final List<SOEnvironment<TransformationSolution>> envs, final String title,
         final int[] maxSteps, final List<JSONObject> conditionList) {
      this.envs = envs;
      this.title = title;
      this.envMaxSteps = new int[envs.size()];
      this.maxFailedRetries = DEFAULT_MAX_FAILED_RETRIES;
      this.failedInARow = new int[envs.size()];
      this.conditionList = conditionList;
      this.archive = new SolutionArchive[envs.size()];
      this.initialSolutions = new TransformationSolution[envs.size()];
      ndp = new NondominatedPopulation();
      hrs = new ResourceSet[envs.size()];
      fWriter = new PrintWriter[envs.size()];
      noExecutions = new AtomicInteger[envs.size()];
      failedCnt = new AtomicInteger[envs.size()];
      ndps = new Population[envs.size()];
      IntStream.range(0, envs.size()).forEach(i -> {

         this.envMaxSteps[i] = maxSteps[i];
         failedCnt[i] = new AtomicInteger(0);
         noExecutions[i] = new AtomicInteger(0);
         hrs[i] = new ResourceSetImpl();
         ndps[i] = new NondominatedPopulation();
         envs.get(i).setSolutionLength(maxSteps[i]);
         this.initialSolutions[i] = envs.get(i).getInitialSolution();
         if(useArchive) {
            this.archive[i] = new SolutionArchive<>();

         }
      });

      final Resource.Factory.Registry reg = Resource.Factory.Registry.INSTANCE;
      reg.getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

   }

   public SingleModelVecEnvironment(final List<SOEnvironment<TransformationSolution>> envs, final String title,
         final int[] maxSteps, final List<JSONObject> conditionList, final int maxFailedRetries) {
      this(envs, title, maxSteps, conditionList);
      if(maxFailedRetries < 1) {
         throw new IllegalArgumentException("maxFailedRetries must be at least 1, got " + maxFailedRetries);
      }
      this.maxFailedRetries = maxFailedRetries;
   }

   protected void addSolutionIfValid(final int envIdx, final Solution s) {
      if(!this.hasNaNOrInfinity(s.getObjectives())) {
         this.ndps[envIdx].add(s);
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
         if(v != v || v == Double.POSITIVE_INFINITY || v == Double.NEGATIVE_INFINITY || Double.isNaN(v)) {
            return true;
         }
      }
      return false;
   }

   protected synchronized boolean isDominatedByAnySolutionInParetoFront(final Solution s) {

      final DominanceComparator comparator = ((NondominatedPopulation) this.ndp).getComparator();
      for(int i = 0; i < this.ndp.size(); i++) {
         // if solution in pareto front dominates solution s, return true
         if(comparator.compare(s, this.ndp.get(i)) > 0) {
            return true;
         }
      }
      return false;
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

      return fSolutions;
   }

   @Handler
   Object[] reset(final int envIdx) {
      final Object[] rv = new Object[2];
      try {
         failedCnt[envIdx].set(0);
         failedInARow[envIdx] = 0;
         noExecutions[envIdx].set(0);

         System.out.println(
               "------------------------------------ INTIALIZED NEW MODEL ------------------------------------");
         if(useArchive && archive[envIdx].size() > 0) {
            final TransformationSolution resetToState = this.archive[envIdx].retrieve(envs.get(envIdx).getEngine());

            envs.get(envIdx).setInitialSolution(resetToState);
            noExecutions[envIdx].set(resetToState.getNumberOfVariables());
            System.out.println("Reset " + envIdx + " to solution of length " + resetToState.getNumberOfVariables());
         } else {
            envs.get(envIdx).setInitialSolution(this.initialSolutions[envIdx]);
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

         if(this.conditionList != null && this.conditionList.get(envIdx) != null) {
            rv[1] = this.conditionList.get(envIdx);
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
      ndp = new NondominatedPopulation();
      hrs = new ResourceSet[envs.size()];
      fWriter = new PrintWriter[envs.size()];
      noExecutions = new AtomicInteger[envs.size()];
      failedCnt = new AtomicInteger[envs.size()];
      ndps = new Population[envs.size()];

      IntStream.range(0, envs.size()).forEach(i -> {
         failedCnt[i] = new AtomicInteger(0);
         failedInARow[i] = 0;
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

   @Handler
   Object[] step(final JSONObject payload) {
      final TransformationSolution realSolution = null;
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

         // A rule Henshin cannot apply is not a transition: no penalty and no budget use.
         // After maxFailedRetries failures in a row, count one step so the env cannot stall.
         if(r.getState() == null) {
            LearnAPIUtils.writeFileAndConsole(fWriter[envIdx], "Failed");
            failedInARow[envIdx] = failedInARow[envIdx] + 1;
            if(failedInARow[envIdx] < this.maxFailedRetries) {
               return new Object[] { false, 0.0, false };
            }
            failedInARow[envIdx] = 0;
            noExecutions[envIdx].incrementAndGet();
            System.out.println("Warning: environment " + envIdx + " failed " + this.maxFailedRetries
                  + " transformations in a row; counting this attempt as one step with reward 0.");
            rv[0] = false;
            rv[1] = 0.0;
         } else {
            failedInARow[envIdx] = 0;
            rv[1] = r.getReward();
            noExecutions[envIdx].incrementAndGet();
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
                  "len: " + r.getState().getSolutionLength() + " (R " + r.getReward() + ")");

            // }
         }
         if(noExecutions[envIdx].get() >= this.envMaxSteps[envIdx]) {
            System.out.println("Reset @ steps = " + noExecutions[envIdx].get());
            rv[2] = true;
         } else {
            System.out.println("current steps = " + noExecutions[envIdx].get());
            rv[2] = false;
         }

         return rv;

      } catch(

      final Exception e) {
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
      final StringBuilder sb = new StringBuilder("\n" + fe + "\n");
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
