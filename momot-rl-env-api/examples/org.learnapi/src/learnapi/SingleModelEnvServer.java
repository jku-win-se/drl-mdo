package learnapi;

import at.ac.tuwien.big.momot.learn.SingleModelVecEnvironment;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.SOEnvironment;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.json.JSONObject;

import co.gongzh.procbridge.Server;

public class SingleModelEnvServer {

   private static Server evalServer;

   public static void main(final String[] args) throws IOException, ParseException {

      final Options options = new Options();
      options.addOption("c", "case", true, "Case ID to run");
      options.addOption("s", "n_steps", true, "");

      options.addOption("n", "n_envs", true, "");
      options.addOption("i", "instance", true, "");
      options.addOption("o", "out_path", true, "");
      options.addOption("p", "port", true, "");

      options.addOption(Option.builder(null).longOpt("test_instances").hasArgs()
            .desc("List of test instances model names").build());

      options.addOption(Option.builder(null).longOpt("test_instances_steps").hasArg()
            .desc("Comma-separated list of steps per test instance").build());

      final CommandLineParser parser = new DefaultParser();
      final CommandLine cmd = parser.parse(options, args);

      final String caseId = String.valueOf(cmd.getOptionValue("case"));
      System.out.println("Running case: " + caseId);
      final int numSteps = Integer.parseInt(cmd.getOptionValue("n_steps"));
      System.out.println("Num. steps per episode: " + numSteps);
      final int numEnvs = Integer.parseInt(cmd.getOptionValue("n_envs"));
      System.out.println("Num. environments: " + numEnvs);
      final String inputModel = String.valueOf(cmd.getOptionValue("instance"));
      System.out.println("Instance: " + inputModel);
      final String outputName = String.valueOf(cmd.getOptionValue("out_path"));
      System.out.println("Out: " + outputName);
      final int port = Integer.valueOf(cmd.getOptionValue("port"));
      System.out.println("Port: " + port);

      List<String> inputModels = null;
      int[] envNumSteps = null;
      List<JSONObject> conditionList = null;

      System.out.println("Output name: " + outputName);
      if(outputName.endsWith("generalization-cra")) {
         inputModels = Collections.nCopies(numEnvs, inputModel);
         envNumSteps = new int[numEnvs];
         Arrays.fill(envNumSteps, Utils.getCRATransformationLength(inputModel + ".xmi"));
      } else if(outputName.endsWith("generalization-slb")) {
         inputModels = Collections.nCopies(numEnvs, inputModel);
         envNumSteps = new int[numEnvs];
         Arrays.fill(envNumSteps, Utils.getSLBTransformationLength(inputModel + ".xmi"));

      } else if(outputName.startsWith("generalization-sp")) {
         inputModels = Collections.nCopies(numEnvs, inputModel);
         envNumSteps = new int[numEnvs];
         Arrays.fill(envNumSteps, Utils.getScrumTransformationLength(inputModel + ".xmi"));
         if(outputName.endsWith("generalization-sp_uc")) {
            conditionList = ConditionUtils.repeat(List.of(Utils.getScrumConditions(inputModel + ".xmi")), numEnvs);
         }
      }
      UseCase uc = UseCase.fromId(caseId, inputModel);

      final List<SOEnvironment<TransformationSolution>> envs = new ArrayList<>();

      for(int i = 0; i < inputModels.size(); i++) {
         uc = UseCase.fromId(caseId, inputModels.get(i));

         final SOEnvironment env = MomotUtil.initSOEnv(uc.moduleManager, uc.function, uc.fitnessAggregator,
               uc.constraintAggregator, uc.fitnessScaler, uc.constraintScaler, null, uc.encodingStrategyClass);

         final TransformationSolution initialSolution = MomotUtil.readInitialSolution(uc.inputGraph,
               uc.function.getObjectiveNames().size(), uc.function.getConstraintNames().size());
         uc.function.evaluate(initialSolution);
         env.setInitialSolution(initialSolution);
         envs.add(env);
      }

      final Server server = new Server(port, new SingleModelVecEnvironment(envs, outputName, envNumSteps, conditionList));
      server.start();
      System.out.println("Training API Server is on " + port + "...");
      evalServer = null;

      Runtime.getRuntime().addShutdownHook(new Thread(() -> {
         System.out.println("Shutdown hook triggered. Cleaning up...");
         server.getListenerThread().interrupt();
         server.stop();
         if(evalServer != null) {
            evalServer.stop();
         }
      }));
   }

}
