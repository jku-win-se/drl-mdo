package at.ac.tuwien.big.moea.search.algorithm;

import at.ac.tuwien.big.moea.ISearchOrchestration;
import at.ac.tuwien.big.moea.search.algorithm.provider.AbstractRegisteredAlgorithm;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.algorithms.NegotiatedWLearning;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.algorithms.ParetoQLearning;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.algorithms.SingleObjectiveQLearning;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.algorithms.WeightedQLearning;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.IEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.IMOEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.ISOEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.EvaluationStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.LocalSearchStrategy;

import java.io.File;
import java.nio.file.Paths;
import java.util.Map;

import org.moeaframework.core.Solution;

public class RLAlgorithmFactory<S extends Solution> extends AbstractAlgorithmFactory<S> {

   protected String outputPath;
   protected S initialSolution;
   protected Map<IEnvironment.Type, IEnvironment<S>> environmentMap;

   // public RLAlgorithmFactory(final ISearchOrchestration<S> searchOrchestration, final List<String>
   // episodeEndingRules,
   // final Map<String, Double> rewardMap, final INeighborhoodFunction<S> neighborhoodFunction,
   // final IFitnessComparator<?, S> fitnessComparator) {
   // setSearchOrchestration(searchOrchestration);
   // this.environment = new Environment<>(this.getInitialSolution(), neighborhoodFunction, fitnessComparator,
   // episodeEndingRules, rewardMap, searchOrchestration.getProblem().getNumberOfVariables());
   //
   // }

   public RLAlgorithmFactory(final ISearchOrchestration<S> searchOrchestration,
         final Map<IEnvironment.Type, IEnvironment<S>> environmentMap) {
      setSearchOrchestration(searchOrchestration);
      for(final IEnvironment<S> env : environmentMap.values()) {
         final S initialSolution = this.getInitialSolution();
         searchOrchestration.getFitnessFunction().evaluate(initialSolution);
         env.setInitialSolution(initialSolution);
         env.setSolutionLength(searchOrchestration.getProblem().getNumberOfVariables());
      }
      this.environmentMap = environmentMap;
      this.outputPath = Paths.get("output", "rl").toString();
      final File f = new File(outputPath).getAbsoluteFile();
      if(!f.exists()) {
         f.mkdirs();
      }

   }

   public AbstractRegisteredAlgorithm<WeightedQLearning<S>> createChebyshevQLearner(final double[] w, final double tau,
         final LocalSearchStrategy localSearchStrategy, final int explorationSteps, final double gamma,
         final double eps, final boolean withEpsDecay, final double epsDecay, final double epsMinimum,
         final String savePath, final int recordInterval, final int terminateAfterEpisodes, final String qTableIn,
         final String qTableOut, final boolean verbose) {

      if(!this.getEnvironments().containsKey(IEnvironment.Type.MOValueBasedEnvironment)) {
         throw new RuntimeException(
               "None of the environments passed to RLAlgorithmFactory is of type  \"MOValueBasedEnvironment\"!");
      }

      return new AbstractRegisteredAlgorithm<WeightedQLearning<S>>() {
         @Override
         public WeightedQLearning<S> createAlgorithm() {
            return new WeightedQLearning<>(w, tau, localSearchStrategy, explorationSteps, gamma, eps, withEpsDecay,
                  epsDecay, epsMinimum, createProblem(), getMOValueEnvironment(),
                  savePath != null ? Paths.get(getOutputPath(), "rewards", savePath).toString() : null, recordInterval,
                  terminateAfterEpisodes, qTableIn, qTableOut, verbose);
         }
      };
   }

   public AbstractRegisteredAlgorithm<ParetoQLearning<S>> createParetoQLearner(
         final LocalSearchStrategy localSearchStrategy, final double[] ref, final int explorationSteps,
         final double gamma, final EvaluationStrategy evaluationStrategy, final double eps, final boolean withEpsDecay,
         final double epsDecay, final double epsMinimum, final String savePath, final int recordInterval,
         final int terminateAfterEpisodes, final String qTableIn, final String qTableOut, final boolean verbose) {

      if(!this.getEnvironments().containsKey(IEnvironment.Type.MOValueBasedEnvironment)) {
         throw new RuntimeException(
               "None of the environments passed to RLAlgorithmFactory is of type  \"MOValueBasedEnvironment\"!");
      }

      return new AbstractRegisteredAlgorithm<ParetoQLearning<S>>() {
         @Override
         public ParetoQLearning<S> createAlgorithm() {
            return new ParetoQLearning<>(localSearchStrategy, ref, explorationSteps, gamma, evaluationStrategy, eps,
                  withEpsDecay, epsDecay, epsMinimum, createProblem(), getMOValueEnvironment(),
                  savePath != null ? Paths.get(getOutputPath(), "rewards", savePath).toString() : null, recordInterval,
                  terminateAfterEpisodes, qTableIn, qTableOut, verbose);
         }
      };
   }

   /**
    * Exploring-focused Q-Learning
    *
    * @param explorationSteps
    *           .. sampling steps in exploration phase
    * @param gamma
    *           .. discount factor
    * @param eps
    *           .. epsilon / probability of entering exploration phase
    * @param withEpsDecay
    *           .. use epsilon decay
    * @param epsDecay
    *           .. epsilon decay value (subtracted from eps when entering exploration phase), if withEpsDecay is used
    * @param epsMinimum
    *           .. minimum epsilon to decay to, if withEpsDecay is used
    * @param savePath
    *           .. storage path
    * @param recordInterval
    *           .. Recording interval / number of epochs
    * @param terminateAfterSeconds
    *           .. If > 0, the training run will terminate after the given amount of time
    * @return
    */
   public AbstractRegisteredAlgorithm<SingleObjectiveQLearning<S>> createSingleObjectiveExploreQLearner(
         final int explorationSteps, final double alpha, final double gamma, final double eps,
         final boolean withEpsDecay, final double epsDecay, final double epsMinimum, final String savePath,
         final int recordInterval, final int terminateAfterEpisodes, final String qTableIn, final String qTableOut,
         final boolean verbose) {

      if(!this.getEnvironments().containsKey(IEnvironment.Type.SOValueBasedEnvironment)) {
         throw new RuntimeException(
               "None of the environments passed to RLAlgorithmFactory is of type  \"SOValueBasedEnvironment\"!");
      }

      return new AbstractRegisteredAlgorithm<SingleObjectiveQLearning<S>>() {
         @Override
         public SingleObjectiveQLearning<S> createAlgorithm() {
            return new SingleObjectiveQLearning<>(LocalSearchStrategy.GREEDY, explorationSteps, alpha, gamma, eps,
                  withEpsDecay, epsDecay, epsMinimum, createProblem(), getSOValueEnvironment(),
                  savePath != null ? Paths.get(getOutputPath(), "rewards", savePath).toString() : null, recordInterval,
                  terminateAfterEpisodes, qTableIn, qTableOut, verbose);
         }
      };
   }

   /**
    * Basic Q-Learning
    *
    * @param gamma
    *           .. discount factor
    * @param eps
    *           .. epsilon / probability of entering exploration phase
    * @param withEpsDecay
    *           .. use epsilon decay
    * @param epsDecay
    *           .. epsilon decay value (subtracted from eps when entering exploration phase), if withEpsDecay is used
    * @param epsMinimum
    *           .. minimum epsilon to decay to, if withEpsDecay is used
    * @param savePath
    *           .. storage path
    * @param recordInterval
    *           .. Recording interval / number of epochs
    * @param terminateAfterSeconds
    *           .. If > 0, the training run will terminate after the given amount of time
    * @return
    */
   public AbstractRegisteredAlgorithm<SingleObjectiveQLearning<S>> createSingleObjectiveQLearner(final double alpha,
         final double gamma, final double eps, final boolean withEpsDecay, final double epsDecay,
         final double epsMinimum, final String savePath, final int recordInterval, final int terminateAfterEpisodes,
         final String qTableIn, final String qTableOut, final boolean verbose) {

      if(!this.getEnvironments().containsKey(IEnvironment.Type.SOValueBasedEnvironment)) {
         throw new RuntimeException(
               "None of the environments passed to RLAlgorithmFactory is of type  \"SOValueBasedEnvironment\"!");
      }

      return new AbstractRegisteredAlgorithm<SingleObjectiveQLearning<S>>() {
         @Override
         public SingleObjectiveQLearning<S> createAlgorithm() {
            return new SingleObjectiveQLearning<>(LocalSearchStrategy.NONE, 1, alpha, gamma, eps, withEpsDecay,
                  epsDecay, epsMinimum, createProblem(), getSOValueEnvironment(),
                  savePath != null ? Paths.get(getOutputPath(), "rewards", savePath).toString() : null, recordInterval,
                  terminateAfterEpisodes, qTableIn, qTableOut, verbose);
         }
      };
   }

   public AbstractRegisteredAlgorithm<NegotiatedWLearning<S>> createTournamentQLearner(
         final LocalSearchStrategy localSearchStrategy, final int explorationSteps, final double gamma,
         final double eps, final boolean withEpsDecay, final double epsDecay, final double epsMinimum,
         final String savePath, final int recordInterval, final int terminateAfterEpisodes, final String qTableIn,
         final String qTableOut, final boolean verbose) {

      if(!this.getEnvironments().containsKey(IEnvironment.Type.MOValueBasedEnvironment)) {
         throw new RuntimeException(
               "None of the environments passed to RLAlgorithmFactory is of type  \"MOValueBasedEnvironment\"!");
      }

      return new AbstractRegisteredAlgorithm<NegotiatedWLearning<S>>() {
         @Override
         public NegotiatedWLearning<S> createAlgorithm() {
            return new NegotiatedWLearning<>(localSearchStrategy, explorationSteps, gamma, eps, withEpsDecay, epsDecay,
                  epsMinimum, createProblem(), getMOValueEnvironment(),
                  savePath != null ? Paths.get(getOutputPath(), "rewards", savePath).toString() : null, recordInterval,
                  terminateAfterEpisodes, qTableIn, qTableOut, verbose);
         }
      };
   }

   // public AbstractRegisteredAlgorithm<SingleObjectivePSQLearning<S>> createSingleObjectiveQLearner(final double
   // gamma,
   // final int resetNoImprSteps, final double eps, final boolean withEpsDecay, final double epsDecay,
   // final double epsMinimum, final String savePath, final int recordInterval, final int terminateAfterSeconds,
   // final String qTableIn, final String qTableOut) {
   //
   //
   // return new AbstractRegisteredAlgorithm<SingleObjectivePSQLearning<S>>() {
   // @Override
   // public SingleObjectivePSQLearning<S> createAlgorithm() {
   // return new SingleObjectivePSQLearning<>(gamma, resetNoImprSteps, eps, withEpsDecay, epsDecay, epsMinimum,
   // createProblem(), getEnvironment(), savePath, recordInterval, terminateAfterSeconds, qTableIn,
   // qTableOut);
   // }
   // };
   // }

   // public IDomainEnvironment<S> getDomainEnvironment() {
   // try {
   // return (IDomainEnvironment<S>) this.environment;
   // } catch(final ClassCastException e) {
   // e.printStackTrace();
   // }
   // return null;
   // }

   public Map<IEnvironment.Type, IEnvironment<S>> getEnvironments() {
      return this.environmentMap;
   }

   public S getInitialSolution() {
      if(initialSolution == null) {
         return getSearchOrchestration().createNewSolution(0);
      }
      return initialSolution;
   }

   public IMOEnvironment<S> getMOValueEnvironment() {
      return (IMOEnvironment<S>) this.getEnvironments().get(IEnvironment.Type.MOValueBasedEnvironment);
   }

   public String getOutputPath() {
      return this.outputPath;
   }

   public ISOEnvironment<S> getPolicyEnvironment() {
      return (ISOEnvironment<S>) this.getEnvironments().get(IEnvironment.Type.PolicyBasedEnvironment);
   }

   // public IGenericEnvironment<S> getGenericEnvironment() {
   // try {
   // return (IGenericEnvironment<S>) this.environment;
   // } catch(final ClassCastException e) {
   // e.printStackTrace();
   // }
   // return null;
   // }

   public ISOEnvironment<S> getSOValueEnvironment() {
      return (ISOEnvironment<S>) this.getEnvironments().get(IEnvironment.Type.SOValueBasedEnvironment);
   }

   public void setInitialSolution(final S initialSolution) {
      this.initialSolution = initialSolution;
   }
}
