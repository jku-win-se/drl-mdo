package at.ac.tuwien.big.moea.search.algorithm.reinforcement.algorithms;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.AbstractSOTabularRLAgent;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.DeltaUpdate;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.ISOEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.LocalSearchStrategy;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import org.eclipse.emf.henshin.interpreter.UnitApplication;
import org.moeaframework.core.Problem;
import org.moeaframework.core.Solution;

public class SingleObjectiveQLearning<S extends Solution> extends AbstractSOTabularRLAgent<S> {

   private final double gamma; // Eagerness - 0 looks in the near future, 1 looks in the distant future
   private double eps;
   private final double epsDecay;
   private final double epsMinimum;
   private final boolean withEpsDecay;
   private final LocalSearchStrategy localSearchStrategy;
   private final int exploreSteps;
   private BufferedWriter bfw;
   private final boolean debug = false;
   private final double alpha;
   private final List<UnitApplication> f = null;

   public SingleObjectiveQLearning(final LocalSearchStrategy localSearchStrategy, final int exploreSteps,
         final double alpha, final double gamma, final double eps, final boolean withEpsDecay, final double epsDecay,
         final double epsMinimum, final Problem problem, final ISOEnvironment<S> environment, final String savePath,
         final int recordInterval, final int terminateAfterEpisodes, final String qTableIn, final String qTableOut,
         final boolean verbose) {
      super(problem, environment, savePath, recordInterval, terminateAfterEpisodes, qTableIn, qTableOut, verbose);
      this.gamma = gamma;
      this.eps = eps;
      this.alpha = alpha;
      this.epsDecay = epsDecay;
      this.epsMinimum = epsMinimum;
      this.withEpsDecay = withEpsDecay;
      this.localSearchStrategy = localSearchStrategy;
      this.exploreSteps = exploreSteps;
      this.bfw = null;
      try {
         this.bfw = new BufferedWriter(new FileWriter("additions.txt", true));
      } catch(final IOException e) {
         // TODO Auto-generated catch block
         e.printStackTrace();
      }

   }

   @Override
   public List<IApplicationState> epsGreedyDecision() {

      List<IApplicationState> nextAction = null;

      if(rng.nextDouble() >= this.eps) {

         // Pick best transformation (max. benefit) for current state
         nextAction = this.qTable.getMaxRewardAction(utils.getApplicationStates(currentSolution));
      }
      return nextAction;
   }

   @Override
   protected void iterate() {
      if(this.startTime == 0) {
         this.startTime = System.currentTimeMillis();
      }

      epochSteps++;
      final List<IApplicationState> nextAction = epsGreedyDecision();

      final SOEnvResponse<S> response = (SOEnvResponse<S>) environment.step(localSearchStrategy, nextAction,
            exploreSteps);

      final DoneStatus doneStatus = response.getDoneStatus();
      final double reward = response.getReward();
      final S nextState = response.getState();

      // if(debug) {
      // System.out.println("len: " + nextState.getNumberOfVariables());
      // System.out.println("rew: " + reward);
      // }
      // final RuleApplication abc = (RuleApplication) nextState.getVariable(nextState.getNumberOfVariables() - 1);
      // int cntspr = 0;
      // if(abc.getCompleteMatch().getUnit().getName().compareTo("moveWorkItemToNewSprintRespectingMax") == 0) {
      // for(int i = 0; i < nextState.getNumberOfVariables(); i++) {
      // final RuleApplication d = (RuleApplication) nextState.getVariable(i);
      // if(d.getCompleteMatch().getUnit().getName().compareTo("moveWorkItemToNewSprintRespectingMax") == 0) {
      // cntspr++;
      // }
      //
      // }
      // }
      // if(nextState == null && nextAction != null
      // && nextAction.get(0).getUnit().getName().compareTo("moveWorkItemToNewSprintRespectingMax") == 0) {
      //
      // System.out.println("TES");
      // }

      if(withEpsDecay) {
         eps = Math.max(epsMinimum, eps - epsDecay * response.getNumExplorationSteps());
      }

      cumReward += reward;

      if(doneStatus != DoneStatus.FINAL_STATE_REACHED) {
         updateQ(utils.getApplicationStates(currentSolution),
               utils.getApplicationStatesDiff(currentSolution, nextState), utils.getApplicationStates(nextState),
               reward);

         addSolutionIfImprovement(nextState);

         iterations++;

         printIfVerboseMode("Iteration: " + iterations);

      }

      if(doneStatus == DoneStatus.MAX_LENGTH_REACHED || doneStatus == DoneStatus.FINAL_STATE_REACHED) {
         if(debug) {

            meanRewards.add(cumReward);
            if(meanRewards.size() > MEAN_REWARD_WINDOW_SIZE) {
               meanRewards.removeFirst();

            }

            System.out.printf("Rolling mean: %.3f (current rew. %.3f, %d iter. consumed)\r",
                  meanRewards.stream().mapToDouble(Double::doubleValue).average().orElse(.0), cumReward, iterations);

         }

         if(this.saveInterval > 0) {
            rewardEarned.add(cumReward);
            framesList.add((double) iterations);
            timePassedList.add((double) (System.currentTimeMillis() - startTime));
            meanRewardEarned.add(cumReward / epochSteps);

            if(terminateAfterEpisodes > 0 && terminateAfterEpisodes <= nrOfEpochs) {
               System.out.println("Terminated after " + terminateAfterEpisodes + " episodes");
               this.terminate();
            }

            if(scoreSavePath != null && nrOfEpochs > 0 && nrOfEpochs % this.saveInterval == 0) {

               printIfVerboseMode("Saving rewards at epoch " + nrOfEpochs + " after "
                     + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

               saveRewards(scoreSavePath, framesList, rewardEarned, timePassedList, meanRewardEarned);
            }

         }

         nrOfEpochs++;
         cumReward = 0;
         epochSteps = 0;
         currentSolution = environment.reset();
         rollbackLog.clear();

      } else {
         currentSolution = nextState;
      }

   }

   private void updateQ(final List<IApplicationState> state, final List<IApplicationState> action,
         final List<IApplicationState> nextState, final double reward) {

      final double transitionReward = this.qTable.getTransitionReward(state, action);

      final boolean hasAddedAsNewState = this.qTable.addStateIfNotExists(nextState);

      final double qUpdateValue = transitionReward
            + alpha * (reward + gamma * this.qTable.getMaxRewardValue(nextState) - transitionReward);

      final double oldValue = this.qTable.getTransitionReward(state, action);
      rollbackLog.add(new DeltaUpdate(state, action, oldValue, hasAddedAsNewState));

      this.qTable.update(state, action, qUpdateValue);

      // try {
      //
      // bfw.write("State " + state.toString() + "\n");
      //
      // bfw.write("Action " + action.toString() + " => " + qUpdateValue);
      // bfw.write("\n\n");
      // bfw.flush();
      //
      // } catch(
      //
      // final IOException e) {
      // // TODO Auto-generated catch block
      // e.printStackTrace();
      // }

   }

}
