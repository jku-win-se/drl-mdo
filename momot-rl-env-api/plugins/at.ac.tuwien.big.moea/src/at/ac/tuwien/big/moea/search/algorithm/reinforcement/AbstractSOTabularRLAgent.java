package at.ac.tuwien.big.moea.search.algorithm.reinforcement;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.DeltaUpdate;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.IApplicationState;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures.ISOQTableAccessor;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.DoneStatus;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.ISOEnvironment;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.SOEnvResponse;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.FileManager;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Problem;
import org.moeaframework.core.Solution;

public abstract class AbstractSOTabularRLAgent<S extends Solution> extends AbstractTabularRLAgent<S> {

   protected static final int MEAN_REWARD_WINDOW_SIZE = 10;

   protected ISOEnvironment<S> environment;

   protected ISOQTableAccessor<List<IApplicationState>, List<IApplicationState>> qTable;
   protected final ArrayList<Double> rewardEarned;
   protected final ArrayList<Double> meanRewardEarned;
   protected final Deque<Double> meanRewards = new ArrayDeque<>(MEAN_REWARD_WINDOW_SIZE);
   protected double cumReward;

   public AbstractSOTabularRLAgent(final Problem problem, final ISOEnvironment<S> environment, final String savePath,
         final int recordInterval, final int terminateAfterEpisodes, final String qTableIn, final String qTableOut,
         final boolean verbose) {
      super(problem, environment, savePath, recordInterval, terminateAfterEpisodes, qTableIn, qTableOut, verbose);

      this.cumReward = 0;
      this.environment = environment;
      this.rewardEarned = new ArrayList<>();
      this.meanRewardEarned = new ArrayList<>();

      if(this.qTableIn != null) {
         this.qTable = this.utils.loadSOQTable(qTableIn, environment.getUnitMapping());
      } else {
         this.qTable = this.utils.initSOQTable(environment.getUnitMapping());
         this.qTable.addStateIfNotExists(new ArrayList<>());
      }
   }

   private NondominatedPopulation exploit() {
      this.qTable.printToFile("qtable1.txt");

      if(rollbackLog.size() > 0) {
         rollbackUpdates();
      }
      this.qTable.printToFile("qtable2.txt");

      currentSolution = environment.reset();
      final NondominatedPopulation p = new NondominatedPopulation();
      int i = 0;
      while(true) {
         final List<IApplicationState> ias = utils.getApplicationStates(currentSolution);

         final List<IApplicationState> nextAction = this.qTable.getMaxRewardAction(ias);
         final SOEnvResponse<S> response = environment.inferStep(nextAction);

         final S nextState = response.getState();
         final DoneStatus ds = response.getDoneStatus();

         if(nextState != null) {
            i++;
            p.add(nextState);
         } else {
            System.out.println("nextState is null");
         }


         if(ds == DoneStatus.MAX_LENGTH_REACHED || ds == DoneStatus.FINAL_STATE_REACHED) {
            break;
         }
         // utils.newTransformationSolution(nextState);

         currentSolution = nextState;
      }

      return p;
   }

   @Override
   public NondominatedPopulation getResult() {
      if(this.qTableOut != null) {
         utils.writeQTableToDisk(this.qTable, this.qTableOut);
      }
      return this.exploit();
   }

   private void rollbackUpdates() {
      for(int i = rollbackLog.size() - 1; i >= 0; i--) {
         final DeltaUpdate u = rollbackLog.get(i);
         if(u.yieldsNewState) {
            this.qTable.removeAction(u.state, u.action);
         } else {
            this.qTable.update(u.state, u.action, u.oldValue);
         }
      }
      rollbackLog.clear();
   }

   public void saveRewards(final String scoreSavePath, final List<Double> framesList, final List<Double> rewardList,
         final List<Double> secondsPassed, final List<Double> meanReward) {
      final ArrayList<ArrayList<Double>> lll = new ArrayList<>();
      lll.add((ArrayList<Double>) framesList);
      lll.add((ArrayList<Double>) rewardList);
      lll.add((ArrayList<Double>) meanReward);
      lll.add((ArrayList<Double>) secondsPassed);
      FileManager.saveBenchMark("evaluations;reward;averageReward;runtime in ms;", lll,
            scoreSavePath + "_" + FileManager.milliSecondsToFormattedDate(this.startTime) + ".csv");
   }

}
