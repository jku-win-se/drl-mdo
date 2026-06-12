package at.ac.tuwien.big.momot.learn;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.BaseEnvironment;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.util.LearnAPIUtils;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.HashMap;
import java.util.IntSummaryStatistics;
import java.util.List;
import java.util.stream.StreamSupport;

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

public class NNDelegateEval extends Delegate {
   private static String BASE_PATH = "output/nn_logs";
   private static String WRITE_PATH = null;
   private static final long timestamp = System.currentTimeMillis() / 1000l;

   private long startTime = 0;
   private final List<Double> runtimes = new ArrayList<>();

   private int noExecutions = 0;
   private int noFailedExecutions = 0;
   private final List<Integer> executions = new ArrayList<>();
   private final List<Integer> failedExecutions = new ArrayList<>();
   private int fe = 0;
   private int failedCnt;
   private PrintWriter fWriterDetails = null;
   private PrintWriter fWriterSummary = null;

   private final List<TransformationSolution> initialSolutions;
   private final String title;

   private int curSolution = 0;
   private int evalPhaseCnt = 0;
   private final HenshinResourceSet hrs = new HenshinResourceSet("");
   private final List<File> fileList;
   private final BaseEnvironment<TransformationSolution> env;
   private final IEGraphMultiDimensionalFitnessFunction realFitnessFunction;
   private Population ndp;
   private final List<Integer> tLengths;

   public NNDelegateEval(final BaseEnvironment<TransformationSolution> env, final String title,
         final IEGraphMultiDimensionalFitnessFunction realFitnessFunction,
         final List<TransformationSolution> initialSolutions, final List<File> fileList, final List<Integer> tLengths) {
      this.env = env;
      this.title = title;
      this.initialSolutions = initialSolutions;
      this.realFitnessFunction = realFitnessFunction;
      this.fileList = fileList;
      this.tLengths = tLengths;
      ndp = new NondominatedPopulation();
      WRITE_PATH = Path.of(BASE_PATH, title).toString();
      failedCnt = 0;
      try {
         Files.createDirectories(Paths.get(WRITE_PATH));
         fWriterSummary = new PrintWriter(new FileWriter(WRITE_PATH + "/evalstats_solutions" + ".txt"));
         fWriterDetails = new PrintWriter(new FileWriter(WRITE_PATH + "/evalstats" + ".txt"));

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
   Object[] isFinished() {

      final boolean allDone = curSolution >= initialSolutions.size();
      if(allDone) {
         LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Finished eval phase " + (++this.evalPhaseCnt));
      }

      final double minVal = StreamSupport.stream(this.ndp.spliterator(), false).mapToDouble(s -> s.getObjective(0))
            .min().orElse(-999);

      final Object[] rv = new Object[2];
      rv[0] = allDone;
      rv[1] = minVal;
      return rv;
   }

   @Handler
   void print() {
      final DoubleSummaryStatistics rStats = runtimes.stream().mapToDouble(Double::doubleValue).summaryStatistics();

      final IntSummaryStatistics eStats = executions.stream().mapToInt(Integer::intValue).summaryStatistics();
      final IntSummaryStatistics efStats = failedExecutions.stream().mapToInt(Integer::intValue).summaryStatistics();
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Runtimes");
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, Arrays.toString(runtimes.toArray()));
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Count: " + rStats.getCount());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Sum: " + rStats.getSum());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Min: " + rStats.getMin());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Average: " + rStats.getAverage());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Max: " + rStats.getMax() + "\n");

      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Executions");
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, Arrays.toString(executions.toArray()));
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Count: " + eStats.getCount());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Sum: " + eStats.getSum());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Min: " + eStats.getMin());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Average: " + eStats.getAverage());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Max: " + eStats.getMax() + "\n");

      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Failed Executions");
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, Arrays.toString(failedExecutions.toArray()));
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Count: " + efStats.getCount());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Sum: " + efStats.getSum());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Min: " + efStats.getMin());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Average: " + efStats.getAverage());
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Max: " + efStats.getMax() + "\n");
   }

   @Handler
   Object[] reset() {
      try {
         noExecutions = 0;
         noFailedExecutions = 0;
         failedCnt = 0;
         // noEpisodes++;
         ndp.clear();
         startTime = System.nanoTime();
         LearnAPIUtils.writeFileAndConsole(fWriterDetails,
               "------------------------------------ INTIALIZED NEW MODEL ------------------------------------");
         LearnAPIUtils.writeFileAndConsole(fWriterDetails, this.fileList.get(curSolution).toString());
         LearnAPIUtils.writeFileAndConsole(fWriterSummary, this.fileList.get(curSolution).toString());

         env.setInitialSolution(initialSolutions.get(curSolution++));
         // env.setInitialSolution(initialSolutions.get(curSolution));

         final TransformationSolution ts = env.reset();
         final Resource res = hrs.createResource(URI.createFileURI("http://MyModel.xmi"));
         res.getContents().addAll(ts.getSourceGraph().getRoots());

         final Object[] rv = new Object[2];
         rv[0] = LearnAPIUtils.transformXMIToString(res);
         return rv;

         // env.getProblemEncoder().printModel(ts);
      } catch(final Exception e) {
         System.out.println(e);
      }
      return null;
   }

   @Handler
   void resetAll() {
      this.curSolution = 0;
      executions.clear();

      failedExecutions.clear();
      runtimes.clear();
   }

   // @Handler
   // void setout(final String dirName) {
   // executions.clear();
   //
   // failedExecutions.clear();
   // runtimes.clear();
   // curSolution = 0;
   //
   // try {
   // Files.createDirectories(Paths.get(WRITE_PATH));
   // fWriterDetails = new PrintWriter(new FileWriter(WRITE_PATH + dirName + ".txt"));
   // } catch(final IOException e) {
   // // TODO Auto-generated catch block
   // e.printStackTrace();
   // }
   // }

   @Handler
   void resetInstance(final String logPath) {
      ndp = new NondominatedPopulation();
      WRITE_PATH = Path.of(BASE_PATH, title, logPath).toString();
      failedCnt = 0;
      try {
         Files.createDirectories(Paths.get(WRITE_PATH));
         fWriterDetails = new PrintWriter(new FileWriter(WRITE_PATH + "/evalstats" + ".txt"));
         fWriterSummary = new PrintWriter(new FileWriter(WRITE_PATH + "/evalstats_solutions" + ".txt"));

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
         LearnAPIUtils.writeFileAndConsole(fWriterDetails, "\nR: " + rule.substring(rule.lastIndexOf("::") + 2));
         paramValues.forEach((key, value) -> LearnAPIUtils.writeFileAndConsole(fWriterDetails, key + " " + value));

         // LearnAPIUtils.writeFileAndConsole(fWriterDetails,System.out.format("%s->%s (%s)", fromId, toId,
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
            failedCnt++;
            LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Failed");
            LearnAPIUtils.writeFileAndConsole(fWriterSummary, "Failed");

         } else {
            failedCnt = 0;
            noExecutions++;

            if(!isDominatedByAnySolutionInParetoFront(r.getState())) {
               final TransformationSolution realSolution = new TransformationSolution(r.getState().getSourceGraph(),
                     r.getState().getVariablesAsList(false), realFitnessFunction.getObjectiveNames().size(),
                     realFitnessFunction.getConstraintNames().size());
               realFitnessFunction.evaluate(realSolution);
               ndp.add(realSolution);

            }

            final Resource res = hrs.createResource(URI.createFileURI("http://MyModel.xmi"));

            res.getContents().addAll(r.getState().getResultGraph().getRoots());

            rv[0] = LearnAPIUtils.transformXMIToString(res);

            // LearnAPIUtils.writeFileAndConsole(fWriterDetails, env.getProblemEncoder().printModel(r.getState()));
            LearnAPIUtils.writeFileAndConsole(fWriterDetails,
                  "len: " + r.getState().getSolutionLength() + " (R " + r.getReward() + ")");

            // }
         }

         rv[2] = r.getDoneStatus() == DoneStatus.FINAL_STATE_REACHED;

         if(noExecutions >= this.tLengths.get(curSolution - 1) || failedCnt >= 5) {
            final double inferTime = (System.nanoTime() - startTime) / Math.pow(10, 9);
            // LearnAPIUtils.writeFileAndConsole(fWriterDetails,inferTime);
            runtimes.add(inferTime);
            executions.add(noExecutions);
            failedExecutions.add(noFailedExecutions);
            writeSolutions(ndp);
            rv[2] = true;

            if(failedCnt >= 5) {
               LearnAPIUtils.writeFileAndConsole(fWriterDetails, "Too many fails, aborted episode");
               LearnAPIUtils.writeFileAndConsole(fWriterSummary, "Too many fails, aborted episode");

            }

         }

         // if(fe % 1000 == 0) {
         // writeSolutions(ndp);
         // }

         return rv;

      } catch(final Exception e) {
         // LearnAPIUtils.writeFileAndConsole(fWriterDetails, e.getMessage());
         throw new RuntimeException(e.getMessage());
      }

      // rv[3] = initialSolutions.size() > curSolution;

      // final OutputStream xmiStr = transformXMIToString(r);
      // return xmiStr;
   }

   private void writeSolutions(final Population p) {
      final StringBuilder sb = new StringBuilder("Transformation solutions:");
      for(final Solution s : p) {
         sb.append(Arrays.toString(s.getObjectives()) + "\n");
      }
      LearnAPIUtils.writeFileAndConsole(fWriterDetails, sb.toString());
      LearnAPIUtils.writeFileAndConsole(fWriterSummary, sb.toString());

   }

}
