package at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IRewardStrategy;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import org.moeaframework.core.Solution;

import container.Container;
import container.ContainerModel;
import container.Stack;

public class ContainerRewardStrategy<S extends Solution> implements IRewardStrategy<S> {

   private double containerIndex(final ContainerModel model) {
      double lowerIdCntBelow = 0;
      final int normalizingConst = IntStream.rangeClosed(1, model.getContainer().size() - 1).sum();
      for(final Stack s : model.getStack()) {
         final int containersOnStackMinus1 = (int) model.getContainer().stream().filter(c -> c.getOn() != null)
               .filter(c -> c.getOn().getId() == s.getId()).count() - 1;
         // final int normalizingConst = containersOnStackMinus1 * (containersOnStackMinus1 + 1) / 2;
         final Container c = s.getTopContainer();
         if(c != null) {
            final List<Integer> stackContainerIds = new ArrayList<>(
                  Arrays.asList(Integer.valueOf(c.getId().substring(1))));
            lowerIdCntBelow += countLowerIdContainersBelowRec(c, normalizingConst, stackContainerIds);
         }
      }
      return lowerIdCntBelow / normalizingConst;
   }

   private double countLowerIdContainersBelowRec(final Container c, final int normalizingConst,
         final List<Integer> stackContainerIds) {
      if(c.getOnTopOf() == null) {
         return 0;
      }
      final Container lowerContainer = c.getOnTopOf();
      final int lowerCId = Integer.valueOf(lowerContainer.getId().substring(1));
      stackContainerIds.add(lowerCId);
      System.out.format("%s = %f containers above\n", lowerContainer.getId(),
            (double) stackContainerIds.stream().filter(id -> id > lowerCId).count());
      return countLowerIdContainersBelowRec(lowerContainer, normalizingConst, stackContainerIds)
            + stackContainerIds.stream().filter(id -> id > lowerCId).count();
   }

   @Override
   public double determineAdditionalReward(final S sOld, final S sNew) {
      final ContainerModel oldModel = MomotUtil.getRoot(((TransformationSolution) sOld).getResultGraph(),
            ContainerModel.class);
      final ContainerModel model = MomotUtil.getRoot(((TransformationSolution) sNew).getResultGraph(),
            ContainerModel.class);
      final boolean hasRetrievedAllContainers = model.getContainer().stream().filter(c -> c.getOn() == null).count()
            - model.getContainer().size() == 0;
      final boolean hasBlockedContainersDecreased = this.containerIndex(model) < this.containerIndex(oldModel);
      final boolean hasRetrievedContainer = model.getContainer().stream().filter(c -> c.getOn() == null)
            .count() > oldModel.getContainer().stream().filter(c -> c.getOn() == null).count();

      // System.out.println("\n" + blockedContainers);
      return hasRetrievedAllContainers ? 20 : !hasRetrievedContainer && hasBlockedContainersDecreased ? 3 : 0;
   }

   @Override
   public Map<String, Double> getRewardMap() {
      final Map<String, Double> rewardMap = new HashMap<>();
      rewardMap.put("retrieveNonLastFromStack", 5.0);
      rewardMap.put("retrieveLastFromStack", 5.0);
      rewardMap.put("retrieveOnTopOfSuccessorFromStack", 5.0);
      rewardMap.put("retrieveLastOverallFromStack", 5.0);
      rewardMap.put("relocateNonLastOnStackToEmptyStack", -1.0);
      rewardMap.put("relocateNonLastOnStackToNonEmptyStack", -1.0);
      rewardMap.put("relocateLastOnStackToNonEmptyStack", -1.0);

      return rewardMap;
   }

}
