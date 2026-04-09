package at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IRewardStrategy;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.util.HashMap;
import java.util.Map;

import org.moeaframework.core.Solution;

import PacmanGame.PositionableEntity;
import PacmanGame.impl.FoodImpl;
import PacmanGame.impl.GameImpl;

public class PacmanRewardStrategy<S extends Solution> implements IRewardStrategy<S> {

   @Override
   public double determineAdditionalReward(final S sOld, final S sNew) {
      final GameImpl game = MomotUtil.getRoot(((TransformationSolution) sNew).getResultGraph(), GameImpl.class);

      boolean foodLeft = false;
      for(final PositionableEntity entity : game.getEntites()) {
         if(entity instanceof FoodImpl) {
            foodLeft = true;
            break;
         }
      }
      if(!foodLeft) {
         return 150;
      }

      return 0;
   }

   @Override
   public Map<String, Double> getRewardMap() {
      final Map<String, Double> rewardMap = new HashMap<>();
      rewardMap.put("eat", 10.0);
      rewardMap.put("kill", -50.0);
      rewardMap.put("moveUp", .0);
      rewardMap.put("moveRight", .0);
      rewardMap.put("moveLeft", .0);
      rewardMap.put("moveDown", .0);
      return rewardMap;
   }

}
