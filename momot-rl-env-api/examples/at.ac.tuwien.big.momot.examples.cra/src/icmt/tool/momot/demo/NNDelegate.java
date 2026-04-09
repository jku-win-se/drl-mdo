package icmt.tool.momot.demo;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.moea.search.fitness.dimension.IFitnessDimension.FunctionType;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.BaseEnvironment;
import at.ac.tuwien.big.momot.util.LearnAPIUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.IntSummaryStatistics;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
import org.json.JSONObject;

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

   private PrintWriter fWriter = null;
   private final List<TransformationSolution> initialSolutions;

   private int curSolution = 0;

   private final HenshinResourceSet hrs = new HenshinResourceSet("");
   private final List<File> fileList;
   private final BaseEnvironment<TransformationSolution> env;

   private final int maxSteps;

   public NNDelegate(final BaseEnvironment<TransformationSolution> env,
         final List<TransformationSolution> initialSolutions, final List<File> fileList, final int maxSteps) {
      this.env = env;
      this.initialSolutions = initialSolutions;
      this.fileList = fileList;
      this.functionType = env.getFunctionType();
      this.maxSteps = maxSteps;
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
      LearnAPIUtils.writeFileAndConsole(fWriter,
            "------------------------------------ INTIALIZED NEW MODEL ------------------------------------");
      LearnAPIUtils.writeFileAndConsole(fWriter, this.fileList.get(curSolution).toString());
      // env.setInitialSolution(initialSolutions.get(curSolution++));
      env.setInitialSolution(initialSolutions.get(curSolution));

      final TransformationSolution ts = env.reset();
      final Resource res = hrs.createResource(URI.createFileURI("http://MyModel.xmi"));
      res.getContents().addAll(ts.getSourceGraph().getRoots());

      final Object[] rv = new Object[2];
      rv[0] = LearnAPIUtils.transformXMIToString(res);

      env.getProblemEncoder().printModel(ts);

      return rv;
   }

   @Handler
   void setout(final String dirName) {
      executions.clear();

      failedExecutions.clear();
      runtimes.clear();
      curSolution = 0;

      try {
         fWriter = new PrintWriter(new FileWriter(WRITE_PATH + dirName + ".txt"));
      } catch(final IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }
   }

   @Handler
   Object[] step(final JSONObject payload) {
      try {
         final String rule = payload.getString("_ruleName");
         final JSONObject paramValues = payload.getJSONObject("_paramValues");

         // LearnAPIUtils.writeFileAndConsole(fWriter,"Amount: " + amount);
         // LearnAPIUtils.writeFileAndConsole(fWriter,System.out.format("%s->%s (%s)", fromId, toId,
         // amount).toString());
         final SOEnvResponse<TransformationSolution> r = env.step(rule, Map.of());
         final Object[] rv = new Object[3];

         // Reward
         rv[1] = r.getReward();

         if(r.getState() == null) {
            // rv[2] = true;
            rv[0] = false;
            rv[1] = -1;
            LearnAPIUtils.writeFileAndConsole(fWriter, "-1");
            noFailedExecutions++;

         } else {
            noExecutions++;
            System.out.println(rule.substring(rule.lastIndexOf("::") + 2));
            LearnAPIUtils.writeFileAndConsole(fWriter, env.getProblemEncoder().printModel(r.getState()));
            System.out.println("L: " + r.getState().getSolutionLength());
            // LearnAPIUtils.writeFileAndConsole(fWriter, String.valueOf((System.nanoTime() - startTime) / Math.pow(10,
            // 9)));
            final Resource res = hrs.createResource(URI.createFileURI("http://MyModel.xmi"));

            res.getContents().addAll(r.getState().getResultGraph().getRoots());

            rv[0] = LearnAPIUtils.transformXMIToString(res);
         }

         rv[2] = r.getDoneStatus() == DoneStatus.FINAL_STATE_REACHED;

         final TransformationSolution nextState = r.getState();
         if(noExecutions >= this.maxSteps) {
            final double inferTime = (System.nanoTime() - startTime) / Math.pow(10, 9);
            // LearnAPIUtils.writeFileAndConsole(fWriter,inferTime);
            runtimes.add(inferTime);
            executions.add(noExecutions);
            failedExecutions.add(noFailedExecutions);

            rv[2] = true;
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
}
