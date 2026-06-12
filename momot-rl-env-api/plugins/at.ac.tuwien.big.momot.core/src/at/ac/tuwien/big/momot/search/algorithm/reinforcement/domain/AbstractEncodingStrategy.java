package at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.moeaframework.core.Solution;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class AbstractEncodingStrategy<S extends Solution> implements IEncodingStrategy<S> {

   @Override
   public INDArray getQValueDistributionForLegalActions(final INDArray dist, final double setVal,
         final List<Integer> legalActions, final int totalActions) {
      final List<Integer> illegalActions = IntStream.rangeClosed(0, totalActions - 1).boxed()
            .collect(Collectors.toList());

      illegalActions.removeAll(legalActions);

      for(final int a : illegalActions) {
         dist.putScalar(a, setVal);
      }
      return dist;

   }

   public int[] integerToOnehot(final int integer, final int bits) {
      final int[] state = new int[bits];
      for(int i = 1; i <= bits; i++) {
         if(i == integer) {
            state[bits - i] = 1;
         } else {
            state[bits - i] = 0;
         }
      }

      return state;
   }

}
