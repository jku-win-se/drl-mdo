package learnapi;

import at.ac.tuwien.big.momot.learn.MultiModelVecEnvironment;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.SOEnvironment;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.json.JSONObject;

import co.gongzh.procbridge.Server;

/**
 * Multi-model vectorized environment server.
 *
 * Serves a training dataset on --port, a validation dataset on port + 1 and a test dataset on port + 2:
 * - Loads .xmi models from a folder, or a single .xmi file, for each dataset
 * - Training envs either stay on one model (env index mod number of models; n_envs must then be a
 *   multiple of the number of models), or cycle through all models (--env_assignment cycle)
 * - Evaluation datasets get --eval_envs envs in total, which the client points at one model at a time
 * - Pre-calculates steps per model based on a case-specific formula
 */
public class MultiModelEnvServer {

   private static MultiModelVecEnvironment createEnvironment(final String caseId, final UseCase uc, final int numEnvs,
         final boolean evalMode, final String outputName, final boolean cycleModels) {
      final List<SOEnvironment<TransformationSolution>> envs = new ArrayList<>();
      final List<IEGraphMultiDimensionalFitnessFunction> envFunctions = new ArrayList<>();

      for(int i = 0; i < numEnvs; i++) {
         final UseCase.EnvConfig envConfig = UseCase.createEnvConfig(caseId, uc.inputModel);
         envs.add(MomotUtil.initSOEnv(envConfig.moduleManager, envConfig.function, envConfig.fitnessAggregator,
               envConfig.constraintAggregator, envConfig.fitnessScaler, envConfig.constraintScaler, null,
               envConfig.encodingStrategyClass));
         envFunctions.add(envConfig.function);
      }

      return new MultiModelVecEnvironment(evalMode, envs, outputName, uc.modelNames, uc.inputGraphs, uc.stepsPerModel,
            envFunctions, uc.conditionsPerModel, uc.fitnessNormalizersPerModel, uc.constraintNormalizersPerModel,
            cycleModels);
   }

   public static void main(final String[] args) throws IOException, ParseException {

      final Options options = new Options();
      options.addOption("c", "case", true, "Case ID to run");
      options.addOption("n", "n_envs", true, "Number of environments (default: 8)");
      options.addOption("m", "models_path", true, "Folder of .xmi models, or one .xmi model");
      options.addOption("o", "out_path", true, "Output path for logs");
      options.addOption("p", "port", true, "Server port");
      options.addOption("v", "val_models_path", true, "Validation dataset, served on port + 1 (default: NONE)");
      options.addOption("t", "test_models_path", true, "Test dataset, served on port + 2 (default: NONE)");
      options.addOption("k", "eval_envs", true, "Environments per evaluation dataset (default: 1)");
      options.addOption("a", "env_assignment", true, "fixed (default) or cycle");

      final CommandLineParser parser = new DefaultParser();
      final CommandLine cmd = parser.parse(options, args);

      final String caseId = String.valueOf(cmd.getOptionValue("case"));
      System.out.println("Running case: " + caseId);

      final int numEnvs = cmd.hasOption("n_envs") ? Integer.parseInt(cmd.getOptionValue("n_envs")) : 8;
      System.out.println("Num. environments: " + numEnvs);

      final String modelsPath = String.valueOf(cmd.getOptionValue("models_path"));
      System.out.println("Models path: " + modelsPath);

      final String outputName = String.valueOf(cmd.getOptionValue("out_path"));
      System.out.println("Out: " + outputName);

      final int port = Integer.valueOf(cmd.getOptionValue("port"));
      System.out.println("Port: " + port);

      final int evalEnvs = Integer.parseInt(cmd.getOptionValue("eval_envs", "1"));

      final String assignment = cmd.getOptionValue("env_assignment", "fixed");
      if(!assignment.equals("fixed") && !assignment.equals("cycle")) {
         throw new IllegalArgumentException("--env_assignment must be fixed or cycle, got: " + assignment);
      }

      final UseCase uc = UseCase.fromIdMultiModel(caseId, modelsPath);

      System.out.println("Loaded " + uc.inputGraphs.size() + " models from " + modelsPath);
      for(int i = 0; i < uc.modelNames.size(); i++) {
         System.out.println("  Model " + i + ": " + uc.modelNames.get(i) + " (steps: " + uc.stepsPerModel.get(i) + ")");
      }

      final MultiModelVecEnvironment delegate = createEnvironment(caseId, uc, numEnvs, false, outputName,
            assignment.equals("cycle"));

      final List<MultiModelVecEnvironment> delegates = new ArrayList<>();
      final List<Server> servers = new ArrayList<>();
      delegates.add(delegate);

      // Evaluation servers start first: a client that reaches the training port finds them ready.
      final JSONObject evalPorts = new JSONObject();
      final JSONObject evalPaths = new JSONObject();
      final String[][] evalDatasets = { { "val", cmd.getOptionValue("val_models_path", "NONE") },
            { "test", cmd.getOptionValue("test_models_path", "NONE") } };
      for(int i = 0; i < evalDatasets.length; i++) {
         final String name = evalDatasets[i][0];
         final String path = evalDatasets[i][1];
         if(path.equals("NONE")) {
            continue;
         }
         final UseCase evalUC = UseCase.fromIdMultiModel(caseId, path);
         final MultiModelVecEnvironment evalDelegate = createEnvironment(caseId, evalUC, evalEnvs, true, outputName,
               false);
         final int evalPort = port + 1 + i;
         final Server evalServer = new Server(evalPort, evalDelegate);
         evalServer.start();
         delegates.add(evalDelegate);
         servers.add(evalServer);
         evalPorts.put(name, evalPort);
         evalPaths.put(name, new File(path).getAbsolutePath());
         System.out.println("Evaluation API Server '" + name + "' (Multi-Model, " + evalEnvs + " envs) is on "
               + evalPort + "...");
      }

      final JSONObject datasets = new JSONObject();
      datasets.put("train", new File(modelsPath).getAbsolutePath());
      final JSONObject runInfo = new JSONObject();
      runInfo.put("case", caseId);
      runInfo.put("eval_ports", evalPorts);
      runInfo.put("datasets", datasets);
      for(final String name : evalPaths.keySet()) {
         datasets.put(name, evalPaths.getString(name));
      }
      delegate.setRunInfo(runInfo);
      final Server server = new Server(port, delegate);
      server.start();
      servers.add(server);
      System.out.println("Training API Server (Multi-Model) is on " + port + "...");

      Runtime.getRuntime().addShutdownHook(new Thread(() -> {
         System.out.println("Shutdown hook triggered. Cleaning up...");
         for(final MultiModelVecEnvironment d : delegates) {
            d.close();
         }
         for(final Server s : servers) {
            s.getListenerThread().interrupt();
            s.stop();
         }
      }));
   }

}
