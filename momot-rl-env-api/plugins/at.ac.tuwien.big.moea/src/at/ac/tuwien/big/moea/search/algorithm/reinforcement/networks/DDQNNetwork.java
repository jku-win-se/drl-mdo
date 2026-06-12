package at.ac.tuwien.big.moea.search.algorithm.reinforcement.networks;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;

public class DDQNNetwork implements INetwork {

   public static INetwork of(final ComputationGraph nn, final String modelSavePath, final int totalActions,
         final double gamma, final boolean enableProgressServer) {
      return new DDQNNetwork(nn, modelSavePath, totalActions, gamma, enableProgressServer);
   }

   private final String modelSavePath;
   private final double gamma;
   private final ComputationGraph nn;

   private final int totalActions;

   private DDQNNetwork(final ComputationGraph nn, final String modelSavePath, final int totalActions,
         final double gamma, final boolean enableProgressServer) {
      this.modelSavePath = modelSavePath;
      this.gamma = gamma;
      this.nn = nn;
      this.totalActions = totalActions;

      if(enableProgressServer) {
         enableServerForTrainingVisualization(nn);
      }

   }

   private void enableServerForTrainingVisualization(final ComputationGraph nn) {
      // Initialize the user interface backend
      System.out.println("Start UI Server ...");
      final UIServer uiServer = UIServer.getInstance();

      // Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
      final StatsStorage statsStorage = new InMemoryStatsStorage(); // Alternative: new FileStatsStorage(File), for
                                                                    // saving and loading later

      uiServer.attach(statsStorage);

      nn.setListeners(new StatsListener(statsStorage));

      System.out.println("Training progress board active!");
   }

   @Override
   public void fit(final ArrayList<INDArray[]> stateArrs, final ArrayList<Integer> action,
         final ArrayList<INDArray> reward) {
      // TODO Auto-generated method stub

   }

   @Override
   public ComputationGraph getComputationGraph() {
      return this.nn;
   }

   @Override
   public double getGamma() {
      return this.gamma;
   }

   @Override
   public INDArray outputSingle(final INDArray[] oldObs) {
      return this.nn.outputSingle(oldObs);
   }

   @Override
   public void saveFinalModel() {
      if(this.modelSavePath != null) {

         try {
            nn.save(new File(modelSavePath + ".mod"));
         } catch(final IOException e) {
            e.printStackTrace();
         }
      }
   }

   @Override
   public void saveModel(final int nrOfEpochs) {
      if(this.modelSavePath != null) {

         try {
            nn.save(new File(modelSavePath + "_" + nrOfEpochs + ".mod"));
         } catch(final IOException e) {
            e.printStackTrace();
         }
      }
   }

}
