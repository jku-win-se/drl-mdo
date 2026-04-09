package learnapi;

import at.ac.tuwien.big.momot.learn.MultiModelVecEnvironment;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.SOEnvironment;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import co.gongzh.procbridge.Server;

/**
 * Multi-model vectorized environment evaluation.
 *
 * Instead of initializing environments with a fixed instance, this version:
 * - Loads multiple .xmi models from a folder (models_path)
 * - Initializes 8 general environments
 * - Cycles through models using round-robin for each reset
 * - Pre-calculates steps per model based on case-specific formula
 */
public class MultiModelEnvServer {

   private static Server evalServer;

   public static void main(final String[] args) throws IOException, ParseException {

      final Options options = new Options();
      options.addOption("c", "case", true, "Case ID to run");
      options.addOption("n", "n_envs", true, "Number of environments (default: 8)");
      options.addOption("m", "models_path", true, "Path to folder containing .xmi models");
      options.addOption("o", "out_path", true, "Output path for logs");
      options.addOption("p", "port", true, "Server port");
      options.addOption("t", "test_models_path", true, "Path to folder containing .xmi models for testing");

      final CommandLineParser parser = new DefaultParser();
      final CommandLine cmd = parser.parse(options, args);

      final String caseId = String.valueOf(cmd.getOptionValue("case"));
      System.out.println("Running case: " + caseId);

      final int numEnvs = cmd.hasOption("n_envs") ? Integer.parseInt(cmd.getOptionValue("n_envs")) : 8;
      System.out.println("Num. environments: " + numEnvs);

      final String modelsPath = String.valueOf(cmd.getOptionValue("models_path"));
      System.out.println("Models path: " + modelsPath);

      final String testModelsPath = String.valueOf(cmd.getOptionValue("test_models_path", "NONE"));
      System.out.println("Test Models path: " + testModelsPath);

      final String outputName = String.valueOf(cmd.getOptionValue("out_path"));
      System.out.println("Out: " + outputName);

      final int port = Integer.valueOf(cmd.getOptionValue("port"));
      System.out.println("Port: " + port);

      final UseCase uc = UseCase.fromIdMultiModel(caseId, modelsPath);

      System.out.println("Loaded " + uc.inputGraphs.size() + " models from " + modelsPath);
      for(int i = 0; i < uc.modelNames.size(); i++) {
         System.out.println("  Model " + i + ": " + uc.modelNames.get(i) + " (steps: " + uc.stepsPerModel.get(i) + ")");
      }

      final List<SOEnvironment<TransformationSolution>> envs = new ArrayList<>();
      final List<at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction> envFunctions = new ArrayList<>();

      for(int i = 0; i < numEnvs; i++) {

         final UseCase.EnvConfig envConfig = UseCase.createEnvConfig(caseId, uc.inputModel);
         final SOEnvironment<TransformationSolution> env = MomotUtil.initSOEnv(envConfig.moduleManager,
               envConfig.function, envConfig.fitnessAggregator, envConfig.constraintAggregator, envConfig.fitnessScaler,
               envConfig.constraintScaler, null, envConfig.encodingStrategyClass);

         envs.add(env);
         envFunctions.add(envConfig.function);
      }
      final MultiModelVecEnvironment delegate = new MultiModelVecEnvironment(false, envs, outputName, uc.modelNames,
            uc.inputGraphs, uc.stepsPerModel, envFunctions, uc.conditionsPerModel, uc.fitnessNormalizersPerModel,
            uc.constraintNormalizersPerModel);

      final Server server = new Server(port, delegate);
      server.start();
      System.out.println("Training API Server (Multi-Model) is on " + port + "...");
      evalServer = null;

      if(testModelsPath.compareTo("NONE") != 0) {
         final UseCase testingUC = UseCase.fromIdMultiModel(caseId, testModelsPath);

         final List<SOEnvironment<TransformationSolution>> testingEnvs = new ArrayList<>();
         final List<at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction> testingEnvFunctions = new ArrayList<>();

         for(int i = 0; i < testingUC.inputGraphs.size(); i++) {

            final UseCase.EnvConfig testingEnvConfig = UseCase.createEnvConfig(caseId, uc.inputModel);
            final SOEnvironment<TransformationSolution> testingEnv = MomotUtil.initSOEnv(testingEnvConfig.moduleManager,
                  testingEnvConfig.function, testingEnvConfig.fitnessAggregator, testingEnvConfig.constraintAggregator,
                  testingEnvConfig.fitnessScaler, testingEnvConfig.constraintScaler, null,
                  testingEnvConfig.encodingStrategyClass);

            testingEnvs.add(testingEnv);
            testingEnvFunctions.add(testingEnvConfig.function);
         }

         final MultiModelVecEnvironment testingDelegate = new MultiModelVecEnvironment(true, testingEnvs, outputName,
               testingUC.modelNames, testingUC.inputGraphs, testingUC.stepsPerModel, testingEnvFunctions,
               testingUC.conditionsPerModel, testingUC.fitnessNormalizersPerModel,
               testingUC.constraintNormalizersPerModel);

         evalServer = new Server(8001, testingDelegate);
         evalServer.start();
         System.out.println("Evaluation API Server (Multi-Model) is on 8001...");

      }

      Runtime.getRuntime().addShutdownHook(new Thread(() -> {
         System.out.println("Shutdown hook triggered. Cleaning up...");
         server.getListenerThread().interrupt();
         server.stop();
         if(evalServer != null) {
            evalServer.getListenerThread().interrupt();
            evalServer.stop();
         }
      }));
   }

}
