package at.ac.tuwien.big.moea.search.algorithm.reinforcement.networks;

import java.util.ArrayList;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface INetwork {
   void fit(final ArrayList<INDArray[]> stateArrs, final ArrayList<Integer> action, final ArrayList<INDArray> reward);

   ComputationGraph getComputationGraph();

   double getGamma();

   INDArray outputSingle(INDArray[] oldObs);

   void saveFinalModel();

   void saveModel(final int nrOfEpochs);
}
