package icmt.tool.momot.demo;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.BaseEnvironment;
import at.ac.tuwien.big.momot.util.LearnAPIUtils;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.IntSummaryStatistics;
import java.util.List;

import javax.annotation.Nullable;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
import org.json.JSONObject;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;
import org.moeaframework.core.comparator.DominanceComparator;

import co.gongzh.procbridge.Delegate;
import co.gongzh.procbridge.Handler;

public class NNDelegate extends Delegate {
   private static final String WRITE_PATH = "output/";

   private FunctionType functionType = null;
   private final int objIndex = 0;

   private long startTime = 0;
   private final List<Double> runtimes = new ArrayList<>();

   private int noExecutions = 0;
   private int noFailedExecutions = 0;
   private final List<Integer> executions = new ArrayList<>();
   private final List<Integer> failedExecutions = new ArrayList<>();
   private int fe = 0;

   private PrintWriter fWriter = null;
   private final List<TransformationSolution> initialSolutions;

   private int curSolution = 0;

   private final HenshinResourceSet hrs = new HenshinResourceSet("");
   private final List<File> fileList;
   private final BaseEnvironment<TransformationSolution> env;

   private final int maxSteps;
   private final Population ndp;
   private final String ts;

   public NNDelegate(final BaseEnvironment<TransformationSolution> env,
         final List<TransformationSolution> initialSolutions, final List<File> fileList, final int maxSteps) {
      this.env = env;
      this.initialSolutions = initialSolutions;
      this.fileList = fileList;
      this.functionType = env.getFunctionType();
      this.maxSteps = maxSteps;
      ndp = new NondominatedPopulation();
      this.ts = String.valueOf(System.currentTimeMillis() / 1000L);
      try {
         Files.createDirectories(Path.of(WRITE_PATH, "nn_log", this.ts));
      } catch(final IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
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

   @Override
   protected @Nullable Object handleUnknownRequest(@Nullable final String method, @Nullable final Object payload) {
      return null;
   }

   protected boolean isDominatedByAnySolutionInParetoFront(final Solution s) {

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
   void print() {
      final DoubleSummaryStatistics rStats = runtimes.stream().mapToDouble(Double::doubleValue).summaryStatistics();

      final IntSummaryStatistics eStats = executions.stream().mapToInt(Integer::intValue).summaryStatistics();
      final IntSummaryStatistics efStats = failedExecutions.stream().mapToInt(Integer::intValue).summaryStatistics();
      LearnAPIUtils.writeFileAndConsole(fWriter, "Runtimes");
      LearnAPIUtils.writeFileAndConsole(fWriter, Arrays.toString(runtimes.toArray()));
      LearnAPIUtils.writeFileAndConsole(fWriter, "Count: " + rStats.getCount());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Sum: " + rStats.getSum());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Min: " + rStats.getMin());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Average: " + rStats.getAverage());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Max: " + rStats.getMax() + "\n");

      LearnAPIUtils.writeFileAndConsole(fWriter, "Executions");
      LearnAPIUtils.writeFileAndConsole(fWriter, Arrays.toString(executions.toArray()));
      LearnAPIUtils.writeFileAndConsole(fWriter, "Count: " + eStats.getCount());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Sum: " + eStats.getSum());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Min: " + eStats.getMin());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Average: " + eStats.getAverage());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Max: " + eStats.getMax() + "\n");

      LearnAPIUtils.writeFileAndConsole(fWriter, "Failed Executions");
      LearnAPIUtils.writeFileAndConsole(fWriter, Arrays.toString(failedExecutions.toArray()));
      LearnAPIUtils.writeFileAndConsole(fWriter, "Count: " + efStats.getCount());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Sum: " + efStats.getSum());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Min: " + efStats.getMin());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Average: " + efStats.getAverage());
      LearnAPIUtils.writeFileAndConsole(fWriter, "Max: " + efStats.getMax() + "\n");
      fWriter.close();
   }

   @Handler
   Object[] reset() {
      noExecutions = 0;
      noFailedExecutions = 0;

      startTime = System.nanoTime();
      System.out
            .println("------------------------------------ INTIALIZED NEW MODEL ------------------------------------");
      System.out.println(this.fileList.get(curSolution).toString());
      // env.setInitialSolution(initialSolutions.get(curSolution++));
      env.setInitialSolution(initialSolutions.get(curSolution));

      final TransformationSolution ts = env.reset();
      final Resource res = hrs.createResource(URI.createFileURI("http://MyModel.xmi"));
      res.getContents().addAll(ts.getSourceGraph().getRoots());

      final Object[] rv = new Object[2];
      rv[0] = LearnAPIUtils.transformXMIToString(res);

      // env.getProblemEncoder().printModel(ts);

      return rv;
   }

   @Handler
   void setout(final String dirName) {
      executions.clear();

      failedExecutions.clear();
      runtimes.clear();
      curSolution = 0;

      try {
         fWriter = new PrintWriter(new FileWriter(WRITE_PATH + dirName + "-" + this.ts + ".txt"));
      } catch(final IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
   }

   @Handler
   Object[] step(final JSONObject payload) {
      try {
         final String rule = payload.getString("_ruleName");
         final HashMap<String, Object> paramValues = new ObjectMapper()
               .readValue(payload.getJSONObject("_paramValues").toString(), HashMap.class);
         // final HashMap<String, Object> paramMap = n
         paramValues.forEach((key, value) -> System.out.println(key + " " + value));

         // LearnAPIUtils.writeFileAndConsole(fWriter,System.out.format("%s->%s (%s)", fromId, toId,
         // amount).toString());
         final SOEnvResponse<TransformationSolution> r = env.step(rule, paramValues);
         final Object[] rv = new Object[3];
         fe++;
         // Reward
         rv[1] = r.getReward();

         if(r.getState() == null) {
            // rv[2] = true;
            rv[0] = false;
            rv[1] = 0;
            noFailedExecutions++;

         } else {
            noExecutions++;
            // System.out.println(rule.substring(rule.lastIndexOf("::") + 2));
            // LearnAPIUtils.writeFileAndConsole(fWriter, env.getProblemEncoder().printModel(r.getState()));
            System.out.println("L: " + r.getState().getSolutionLength() + " (CRAindex " + r.getReward() + ")");
            // LearnAPIUtils.writeFileAndConsole(fWriter, String.valueOf((System.nanoTime() - startTime) / Math.pow(10,
            // 9)));

            if(!isDominatedByAnySolutionInParetoFront(r.getState())) {
               ndp.add(r.getState());
               LearnAPIUtils.writeFileAndConsole(fWriter,
                     "new Solution: " + Arrays.toString(r.getState().getObjectives()));
            }

            final Resource res = hrs.createResource(URI.createFileURI("http://MyModel.xmi"));

            res.getContents().addAll(r.getState().getResultGraph().getRoots());

            rv[0] = LearnAPIUtils.transformXMIToString(res);
         }

         rv[2] = r.getDoneStatus() == DoneStatus.FINAL_STATE_REACHED;

         if(noExecutions >= this.maxSteps) {
            final double inferTime = (System.nanoTime() - startTime) / Math.pow(10, 9);
            // LearnAPIUtils.writeFileAndConsole(fWriter,inferTime);
            runtimes.add(inferTime);
            executions.add(noExecutions);
            failedExecutions.add(noFailedExecutions);

            rv[2] = true;
         }

         if(fe % 1000 == 0) {
            writeSolutions(ndp);
         }

         return rv;

      } catch(final Exception e) {
         LearnAPIUtils.writeFileAndConsole(fWriter, e.getMessage());
      }
      return null;

      // rv[3] = initialSolutions.size() > curSolution;

      // final OutputStream xmiStr = transformXMIToString(r);
      // return xmiStr;
   }

   private void writeSolutions(final Population p) {
      final StringBuilder sb = new StringBuilder();
      for(final Solution s : p) {
         sb.append(Arrays.toString(s.getObjectives()) + "\n");
      }

      try(PrintWriter out = new PrintWriter(
            Path.of(WRITE_PATH, "nn_log", this.ts, String.valueOf(fe)).toString() + ".txt")) {
         out.println(sb.toString());
      } catch(final FileNotFoundException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
   }

}
