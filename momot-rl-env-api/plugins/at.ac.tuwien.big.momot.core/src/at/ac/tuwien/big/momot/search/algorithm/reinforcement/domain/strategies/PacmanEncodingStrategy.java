package at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.strategies;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.FixedRuleApplicationStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.UnitParameter;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.networks.CustomLoss;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.AbstractEncodingStrategy;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.moeaframework.core.Solution;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import PacmanGame.Game;
import PacmanGame.GridNode;
import PacmanGame.PositionableEntity;
import PacmanGame.impl.FoodImpl;
import PacmanGame.impl.GameImpl;
import PacmanGame.impl.GhostImpl;
import PacmanGame.impl.PacmanImpl;

public class PacmanEncodingStrategy<S extends Solution> extends AbstractEncodingStrategy<S> {

   private static int[] oneHotNearFoodVector(final List<Point> foodPos, final Point pacPos) {
      final int pacY = (int) pacPos.getY();
      final int pacX = (int) pacPos.getX();

      final int[] oneHotFoodVector = new int[4];
      if(foodPos.contains(new Point(pacX, pacY - 1))) {
         oneHotFoodVector[0] = 1;
      }
      if(foodPos.contains(new Point(pacX + 1, pacY))) {
         oneHotFoodVector[1] = 1;
      }
      if(foodPos.contains(new Point(pacX, pacY + 1))) {
         oneHotFoodVector[2] = 1;
      }
      if(foodPos.contains(new Point(pacX - 1, pacY))) {
         oneHotFoodVector[3] = 1;
      }
      return oneHotFoodVector;
   }

   private static int[] oneHotNearGhostVector(final List<Point> ghostPos, final Point pacPos) {
      final int pacY = (int) pacPos.getY();
      final int pacX = (int) pacPos.getX();

      final int[] oneHotGhostVector = new int[4];
      if(ghostPos.contains(new Point(pacX, pacY - 1))) {
         oneHotGhostVector[0] = 1;
      }
      if(ghostPos.contains(new Point(pacX + 1, pacY))) {
         oneHotGhostVector[1] = 1;
      }
      if(ghostPos.contains(new Point(pacX, pacY + 1))) {
         oneHotGhostVector[2] = 1;
      }
      if(ghostPos.contains(new Point(pacX - 1, pacY))) {
         oneHotGhostVector[3] = 1;
      }
      return oneHotGhostVector;
   }

   @Override
   public List<UnitParameter> createBaseRules() {
      final List<UnitParameter> baseRules = new ArrayList<>();
      Map<String, Object> parameterValues = new HashMap<>();

      parameterValues.put("mover", "p1");
      baseRules.add(new UnitParameter("move_up::moveUp", parameterValues));

      parameterValues = new HashMap<>();
      parameterValues.put("mover", "p1");
      baseRules.add(new UnitParameter("move_right::moveRight", parameterValues));

      parameterValues = new HashMap<>();
      parameterValues.put("mover", "p1");
      baseRules.add(new UnitParameter("move_down::moveDown", parameterValues));

      parameterValues = new HashMap<>();
      parameterValues.put("mover", "p1");
      baseRules.add(new UnitParameter("move_left::moveLeft", parameterValues));

      return baseRules;
   }

   @Override
   public List<UnitParameter> createPostBaseRules() {
      final List<UnitParameter> postBaseRules = new ArrayList<>();
      final Map<String, Object> parameterValues = new HashMap<>();

      postBaseRules.add(new UnitParameter("eat::eat", parameterValues));
      postBaseRules.add(new UnitParameter("kill::kill", parameterValues));

      return postBaseRules;
   }

   @Override
   public INDArray[] encodeSolution(final S s) {
      final TransformationSolution solution = (TransformationSolution) s;
      final INDArray[] encodingsArr = new INDArray[1];

      final GameImpl game = MomotUtil.getRoot(solution.getResultGraph(), GameImpl.class);

      final int boardHeight = (int) Math.sqrt(game.getGridnodes().size());
      final int boardWidth = boardHeight;
      // Per gridnode of game, assign integer for node state
      // 0 = nothing, 1 = pacman, 2 = food, 3 = ghost
      final INDArray encoding = Nd4j.zeros(1, 9);

      final Map<String, Integer> nodeStates = new HashMap<>();
      for(final GridNode g : game.getGridnodes()) {
         nodeStates.put(g.getId(), 0);

      }
      final Point pacPos = new Point();

      final List<Point> foodPos = new ArrayList<>();
      final List<Point> ghostPos = new ArrayList<>();

      double foodPiecesAtStart = 0;

      final GameImpl startGame = MomotUtil.getRoot(solution.getSourceGraph(), GameImpl.class);

      for(final PositionableEntity entity : startGame.getEntites()) {
         if(entity instanceof FoodImpl) {
            foodPiecesAtStart++;
         }
      }

      for(final PositionableEntity entity : game.getEntites()) {
         if(entity instanceof FoodImpl) {
            final String foodId = ((FoodImpl) entity).getOn().getId();
            foodPos.add(new Point(Integer.parseInt(foodId.split("_")[1]), Integer.parseInt(foodId.split("_")[0])));
         } else if(entity instanceof GhostImpl) {
            final String ghostId = ((GhostImpl) entity).getOn().getId();
            ghostPos.add(new Point(Integer.parseInt(ghostId.split("_")[1]), Integer.parseInt(ghostId.split("_")[0])));

         } else if(entity instanceof PacmanImpl) {
            final String pacId = ((PacmanImpl) entity).getOn().getId();
            pacPos.setLocation(Integer.parseInt(pacId.split("_")[1]), Integer.parseInt(pacId.split("_")[0]));

         }
      }

      // for(final PositionableEntity entity : game.getEntites()) {
      // if(entity instanceof PacmanImpl) {
      // final String pacId = ((PacmanImpl) entity).getOn().getId();
      // nodeStates.put(pacId, 1);
      //
      // pacPos.setLocation(Integer.parseInt(pacId.split("_")[1]), Integer.parseInt(pacId.split("_")[0]));
      //
      // } else if(entity instanceof FoodImpl) {
      // final String foodId = ((FoodImpl) entity).getOn().getId();
      //
      // nodeStates.put(foodId, 2);
      //
      // foodPos.add(new Point(Integer.parseInt(foodId.split("_")[1]), Integer.parseInt(foodId.split("_")[0])));
      //
      // } else if(entity instanceof GhostImpl) {
      // final String ghostId = ((GhostImpl) entity).getOn().getId();
      //
      // nodeStates.put(ghostId, 3);
      // ghostPos.add(new Point(Integer.parseInt(ghostId.split("_")[1]), Integer.parseInt(ghostId.split("_")[0])));
      //
      // }
      // }
      //
      // final List<String> sortedNodeIds = new ArrayList<>(nodeStates.keySet());
      // Collections.sort(sortedNodeIds);
      int column = 0;
      // for(final String nodeId : sortedNodeIds) {
      // final int[] oneHotState = integerToOnehot(nodeStates.get(nodeId), 3);
      // for(final int i : oneHotState) {
      // encoding.put(0, column++, i);
      // }
      //
      // }

      // final List<Map.Entry<String, Integer>> sortedList = nodeStates.entrySet().stream()
      // .sorted(Map.Entry.comparingByKey()).collect(Collectors.toList());

      // final int cnt = 1;

      // for(final Map.Entry<String, Integer> e : sortedList) {
      // System.out.print(e.getValue());
      // if(cnt % Math.sqrt(game.getGridnodes().size()) == 0) {
      // System.out.println();
      // }
      // cnt++;
      // }

      // final double[] pacmanGhostNearnessVector = ghostDistanceVector(ghostPos, pacPos, boardWidth, boardHeight);
      // for(final double i : pacmanGhostNearnessVector) {
      // encoding.put(0, column++, i);
      // }

      // final int[] pacmanNearFoodVector = oneHotNearFoodVector(foodPos, pacPos);
      // for(final double i : pacmanNearFoodVector) {
      // encoding.put(0, column++, i);
      // }
      // System.out.println("near food: " + Arrays.toString(pacmanNearFoodVector));

      final int[] pacmanNearGhostVector = oneHotNearGhostVector(ghostPos, pacPos);
      for(final double i : pacmanNearGhostVector) {
         encoding.put(0, column++, i);
      }
      System.out.println("near ghost: " + Arrays.toString(pacmanNearGhostVector));

      // final double[] ghostDistanceVector = ghostDistanceVector(ghostPos, pacPos, boardWidth, boardHeight);
      // for(final double i : ghostDistanceVector) {
      // encoding.put(0, column++, i);
      // }

      final double[] pacmanFoodNearnessVector = foodDistanceVector(foodPos, pacPos, boardWidth, boardHeight);
      for(final double i : pacmanFoodNearnessVector) {
         encoding.put(0, column++, i);
      }
      final double foodProgress = (foodPiecesAtStart - foodPos.size()) / foodPiecesAtStart;
      System.out.println("Food progress: " + foodProgress);
      encoding.putScalar(0, column, foodProgress);

      // System.out.println(Arrays.toString(pacmanFoodNearnessVector));

      encodingsArr[0] = encoding;
      return encodingsArr;
   }

   private double[] foodDistanceVector(final List<Point> foodPos, final Point pacPos, final double boardWidth,
         final double boardHeight) {
      final int pacY = (int) pacPos.getY();
      final int pacX = (int) pacPos.getX();

      double yUpNearness = 0;
      double yDownNearness = 0;
      double xRightNearness = 0;
      double xLeftNearness = 0;

      for(final Point food : foodPos) {
         if(food.y == pacY) {
            if(food.x < pacX) {
               final double curXLeftXNearness = 1 - (Math.abs(pacX - food.x) - 1) / boardWidth;
               if(curXLeftXNearness > xLeftNearness) {
                  xLeftNearness = curXLeftXNearness;
               }
            } else if(food.x > pacX) {
               final double curXRightNearness = 1 - (Math.abs(pacX - food.x) - 1) / boardWidth;
               if(curXRightNearness > xRightNearness) {
                  xRightNearness = curXRightNearness;
               }
            }
         } else if(food.x == pacX) {
            if(food.y < pacY) {
               final double curYUpNearness = 1 - (Math.abs(pacY - food.y) - 1) / boardHeight;

               if(curYUpNearness > yUpNearness) {
                  yUpNearness = curYUpNearness;
               }
            } else if(food.y > pacY) {
               final double curYDownNearness = 1 - (Math.abs(pacY - food.y) - 1) / boardHeight;
               if(curYDownNearness > yDownNearness) {
                  yDownNearness = curYDownNearness;
               }
            }
         }

      }

      final double[] nearnessVector = new double[4];
      nearnessVector[0] = yUpNearness;
      nearnessVector[1] = xRightNearness;
      nearnessVector[2] = yDownNearness;
      nearnessVector[3] = xLeftNearness;

      System.out.println("Nearness (F): " + Arrays.toString(nearnessVector));
      return nearnessVector;
   }

   @Override
   public int getActionSpace(final S initialSolution) {
      return 4;
   }

   @Override
   public ComputationGraph getActorArchitecture(final S solution, final double lr, final double l2Regularization,
         final double dropoutRate, final boolean printArchitecture) {
      final ComputationGraphConfiguration.GraphBuilder confBuild = new NeuralNetConfiguration.Builder().seed(123)
            .updater(new Adam(lr)).weightInit(WeightInit.NORMAL).l2(l2Regularization).dropOut(1 - dropoutRate)
            .graphBuilder().addInputs("disc_in");

      final InputType[] outputTypes = new InputType[1];
      outputTypes[0] = InputType.feedForward(this.getStateSpace(solution));

      confBuild.addLayer("policy_1", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(256).build(),
            "disc_in");
      confBuild.addLayer("policy_2", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(256).build(),
            "policy_1");

      confBuild.addLayer("action", new OutputLayer.Builder(new CustomLoss()).nOut(this.getActionSpace(solution))
            .activation(Activation.SOFTMAX).build(), "policy_2");

      confBuild.setOutputs("action").setInputTypes(outputTypes).build();

      final ComputationGraph net = new ComputationGraph(confBuild.build());
      net.init();

      return net;
   }

   @Override
   public ComputationGraph getCriticArchitecture(final S solution, final double lr, final double l2Regularization,
         final double dropoutRate, final boolean printArchitecture) {
      final ComputationGraphConfiguration.GraphBuilder confBuild = new NeuralNetConfiguration.Builder().seed(123)
            .updater(new Adam(lr)).weightInit(WeightInit.NORMAL).l2(l2Regularization).dropOut(1 - dropoutRate)
            .graphBuilder().addInputs("disc_in");

      final InputType[] outputTypes = new InputType[1];
      outputTypes[0] = InputType.feedForward(this.getStateSpace(solution));

      confBuild.addLayer("policy_1", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(128).build(),
            "disc_in");
      confBuild.addLayer("policy_2", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(64).build(),
            "policy_1");

      confBuild.addLayer("action",
            new OutputLayer.Builder(new CustomLoss()).nOut(1).activation(Activation.IDENTITY).build(), "policy_2");

      confBuild.setOutputs("action").setInputTypes(outputTypes).build();

      final ComputationGraph net = new ComputationGraph(confBuild.build());
      net.init();

      return net;
   }

   @Override
   public ComputationGraph getDDQNArchitecture(final S solution, final double learningRate,
         final double l2Regularization, final double dropoutRate, final boolean verbose) {
      final ComputationGraphConfiguration.GraphBuilder confBuild = new NeuralNetConfiguration.Builder().seed(123)
            .updater(new Adam(learningRate)).weightInit(WeightInit.XAVIER).l2(l2Regularization).dropOut(1 - dropoutRate)
            .graphBuilder().addInputs("disc_in");

      final InputType[] outputTypes = new InputType[1];
      outputTypes[0] = InputType.feedForward(this.getStateSpace(solution));

      confBuild.addLayer("policy_1", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(300).build(),
            "disc_in");
      confBuild.addLayer("policy_2", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(300).build(),
            "policy_1");

      confBuild.addLayer("action", new OutputLayer.Builder(LossFunctions.LossFunction.L2)
            .nOut(this.getActionSpace(solution)).activation(Activation.IDENTITY).build(), "policy_2");

      confBuild.setOutputs("action").setInputTypes(outputTypes).build();

      final ComputationGraph net = new ComputationGraph(confBuild.build());
      net.init();

      return net;
   }

   @Override
   public String getDistrActionProbabilities(INDArray dist, final S solution, final boolean renormalizeIllegalActions) {
      final Game game = MomotUtil.getRoot(((TransformationSolution) solution).getResultGraph(), Game.class);

      // if "from" stack is empty, cannot relocate from it
      final Optional<PositionableEntity> oe = game.getEntites().stream().filter(e -> e instanceof PacmanImpl)
            .findFirst();
      final int row = Integer.valueOf(oe.get().getOn().getId().split("_")[0]);
      final int col = Integer.valueOf(oe.get().getOn().getId().split("_")[1]);

      final int boardHeight = (int) Math.sqrt(game.getGridnodes().size());
      final int boardWidth = boardHeight;

      if(row == 0) {
         dist.putScalar(0, 0.0);
      }
      if(row == boardHeight - 1) {
         dist.putScalar(2, 0.0);
      }

      if(col == 0) {
         dist.putScalar(3, 0.0);
      }
      if(col == boardWidth - 1) {
         dist.putScalar(1, 0.0);
      }
      if(renormalizeIllegalActions) {

         dist = dist.div(dist.sum(1));
      }

      final StringBuilder sb = new StringBuilder();
      final double[] dDist = dist.toDoubleVector();

      sb.append(String.format("Up => %.3f | Right => %.3f | Down => %.3f | Left => %.3f", dDist[0], dDist[1], dDist[2],
            dDist[3]));
      return sb.toString();
   }

   @Override
   public List<String> getEpisodeEndingRules() {
      final List<String> endingRules = new ArrayList<>();
      endingRules.add(new String("kill"));
      return endingRules;
   }

   @Override
   public FixedRuleApplicationStrategy getFixedUnitApplicationStrategy(final S s, final int action) {
      final FixedRuleApplicationStrategy applicationStrategy = new FixedRuleApplicationStrategy();
      String moveUnitName = null;
      final Map<String, Object> parameterValues = new HashMap<>();
      UnitParameter moveAction = null;

      // 0 = up, 1 = right, 2 = down, 3 = left

      // 1. main role (move step in one direction)
      moveUnitName = null;

      switch(action) {
         case 0:
            moveUnitName = "move_up::moveUp";
            parameterValues.put("mover", "p1"); // p1 = pacman
            break;
         case 1:
            moveUnitName = "move_right::moveRight";
            parameterValues.put("mover", "p1"); // p1 = pacman
            break;
         case 2:
            moveUnitName = "move_down::moveDown";
            parameterValues.put("mover", "p1"); // p1 = pacman
            break;
         case 3:
            moveUnitName = "move_left::moveLeft";
            parameterValues.put("mover", "p1"); // p1 = pacman
            break;

      }
      moveAction = new UnitParameter(moveUnitName, parameterValues);
      applicationStrategy.setDistributionSampleRule(moveAction);

      // Subsequent optional rules: Check if can apply rule "eat"
      final List<UnitParameter> optionalSubsequentRules = new ArrayList<>();
      final Map<String, Object> subseqRuleParameterValues = new HashMap<>();
      // subseqRuleParameterValues.put("score_value", "1");
      optionalSubsequentRules.add(new UnitParameter("eat::eat", subseqRuleParameterValues));
      optionalSubsequentRules.add(new UnitParameter("kill::kill", subseqRuleParameterValues));

      applicationStrategy.setOptionalSubsequentRules(optionalSubsequentRules);

      return applicationStrategy;

   }

   @Override
   public List<Integer> getLegalActions(final S solution) {
      final List<Integer> legalActions = IntStream.rangeClosed(0, this.getActionSpace(solution) - 1).boxed()
            .collect(Collectors.toList());

      final Game game = MomotUtil.getRoot(((TransformationSolution) solution).getResultGraph(), Game.class);

      // if "from" stack is empty, cannot relocate from it
      final Optional<PositionableEntity> oe = game.getEntites().stream().filter(e -> e instanceof PacmanImpl)
            .findFirst();

      throw new RuntimeException("todo compatibility comment");
      // if(oe.isEmpty()) {
      // return List.of();
      // }

      // final int row = Integer.valueOf(oe.get().getOn().getId().split("_")[0]);
      // final int col = Integer.valueOf(oe.get().getOn().getId().split("_")[1]);
      //
      // System.out.println(row);
      // System.out.println(col);
      //
      // final int boardHeight = (int) Math.sqrt(game.getGridnodes().size());
      // final int boardWidth = boardHeight;
      //
      // if(row == 0) {
      // legalActions.remove(Integer.valueOf(0));
      // }
      // if(row == boardHeight - 1) {
      // legalActions.remove(Integer.valueOf(2));
      // }
      //
      // if(col == 0) {
      // legalActions.remove(Integer.valueOf(3));
      // }
      // if(col == boardWidth - 1) {
      // legalActions.remove(Integer.valueOf(1));
      // }
      //
      // return legalActions;

   }

   @Override
   public String getModelVisualization(final S s) {
      final GameImpl game = MomotUtil.getRoot(((TransformationSolution) s).getResultGraph(), GameImpl.class);
      final Map<String, String> fieldToChar = new HashMap<>();

      final Map<String, String> nodeStates = new HashMap<>();
      for(final GridNode g : game.getGridnodes()) {
         nodeStates.put(g.getId(), "-");

      }

      for(final PositionableEntity entity : game.getEntites()) {
         if(entity instanceof PacmanImpl) {
            final String pacId = ((PacmanImpl) entity).getOn().getId();
            nodeStates.put(pacId, "P");

         } else if(entity instanceof FoodImpl) {
            final String foodId = ((FoodImpl) entity).getOn().getId();

            nodeStates.put(foodId, "F");

         } else if(entity instanceof GhostImpl) {
            final String ghostId = ((GhostImpl) entity).getOn().getId();

            nodeStates.put(ghostId, "G");

         }
      }

      final List<String> sortedNodeIds = new ArrayList<>(nodeStates.keySet());

      final List<Entry<String, String>> sortedList = nodeStates.entrySet().stream().sorted(Map.Entry.comparingByKey())
            .collect(Collectors.toList());

      int cnt = 1;

      final StringBuilder sb = new StringBuilder();
      for(final Entry<String, String> e : sortedList) {
         sb.append(e.getValue());
         if(cnt % Math.sqrt(game.getGridnodes().size()) == 0) {
            sb.append("\n");
         }
         cnt++;
      }
      return sb.toString();
   }

   @Override
   public ComputationGraph getNetworkArchitecture(final S solution, final double lr, final double l2Regularization,
         final double dropoutRate, final boolean printArchitecture) {
      final ComputationGraphConfiguration.GraphBuilder confBuild = new NeuralNetConfiguration.Builder().seed(123)
            .updater(new Adam(lr)).weightInit(WeightInit.NORMAL).l2(l2Regularization).dropOut(1 - dropoutRate)
            .graphBuilder().addInputs("disc_in");

      final InputType[] outputTypes = new InputType[1];
      outputTypes[0] = InputType.feedForward(this.getStateSpace(solution));

      confBuild.addLayer("policy_1", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(128).build(),
            "disc_in");
      confBuild.addLayer("policy_2", new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(64).build(),
            "policy_1");

      confBuild.addLayer("action", new OutputLayer.Builder(new CustomLoss()).nOut(this.getActionSpace(solution))
            .activation(Activation.SOFTMAX).build(), "policy_2");

      confBuild.setOutputs("action").setInputTypes(outputTypes).build();

      final ComputationGraph net = new ComputationGraph(confBuild.build());
      net.init();

      return net;
   }

   @Override
   public int getStateSpace(final S initialSolution) {
      final TransformationSolution solution = (TransformationSolution) initialSolution;
      final Game game = MomotUtil.getRoot(solution.execute(), Game.class);
      return 9;
   }

   private String getTextBefore(final String wholeString, final String before) {
      final int indexOf = wholeString.indexOf(before);
      if(indexOf != -1) {
         return wholeString.substring(0, indexOf);
      }
      return wholeString;
   }

   private double[] ghostDistanceVector(final List<Point> ghostPos, final Point pacPos, final double boardWidth,
         final double boardHeight) {
      final int pacY = (int) pacPos.getY();
      final int pacX = (int) pacPos.getX();

      double yUpNearness = 0;
      double yDownNearness = 0;
      double xRightNearness = 0;
      double xLeftNearness = 0;

      for(final Point food : ghostPos) {
         if(food.y == pacY) {
            if(food.x < pacX) {
               final double curXLeftXNearness = 1 - (Math.abs(pacX - food.x) - 1) / boardWidth;
               if(curXLeftXNearness > xLeftNearness) {
                  xLeftNearness = curXLeftXNearness;
               }
            } else if(food.x > pacX) {
               final double curXRightNearness = 1 - (Math.abs(pacX - food.x) - 1) / boardWidth;
               if(curXRightNearness > xRightNearness) {
                  xRightNearness = curXRightNearness;
               }
            }
         } else if(food.x == pacX) {
            if(food.y < pacY) {
               final double curYUpNearness = 1 - (Math.abs(pacY - food.y) - 1) / boardHeight;

               if(curYUpNearness > yUpNearness) {
                  yUpNearness = curYUpNearness;
               }
            } else if(food.y > pacY) {
               final double curYDownNearness = 1 - (Math.abs(pacY - food.y) - 1) / boardHeight;
               if(curYDownNearness > yDownNearness) {
                  yDownNearness = curYDownNearness;
               }
            }
         }

      }

      final double[] nearnessVector = new double[4];
      nearnessVector[0] = yUpNearness;
      nearnessVector[1] = xRightNearness;
      nearnessVector[2] = yDownNearness;
      nearnessVector[3] = xLeftNearness;

      System.out.println("Nearness (G): " + Arrays.toString(nearnessVector));
      return nearnessVector;
   }

   @Override
   public boolean isEpisodeEndingState(final S s) {
      final GameImpl game = MomotUtil.getRoot(((TransformationSolution) s).getResultGraph(), GameImpl.class);

      boolean foodLeft = false;
      boolean pacmanAlive = false;
      for(final PositionableEntity entity : game.getEntites()) {
         if(entity instanceof FoodImpl) {
            foodLeft = true;
         } else if(entity instanceof PacmanImpl) {
            pacmanAlive = true;
         }

         if(foodLeft && pacmanAlive) {
            break;
         }

      }

      if(!foodLeft || !pacmanAlive) {
         return true;
      }
      return false;
   }

   @Override
   public boolean isFinalModel(final S s) {
      // TODO Auto-generated method stub
      return false;
   }

   @Override
   public String printModel(final S s) {
      final GameImpl game = MomotUtil.getRoot(((TransformationSolution) s).getResultGraph(), GameImpl.class);
      final Map<String, String> fieldToChar = new HashMap<>();

      final Map<String, String> nodeStates = new HashMap<>();
      for(final GridNode g : game.getGridnodes()) {
         nodeStates.put(g.getId(), "-");

      }

      for(final PositionableEntity entity : game.getEntites()) {
         if(entity instanceof PacmanImpl) {
            final String pacId = ((PacmanImpl) entity).getOn().getId();
            nodeStates.put(pacId, "P");

         } else if(entity instanceof FoodImpl) {
            final String foodId = ((FoodImpl) entity).getOn().getId();

            nodeStates.put(foodId, "F");

         } else if(entity instanceof GhostImpl) {
            final String ghostId = ((GhostImpl) entity).getOn().getId();

            nodeStates.put(ghostId, "G");

         }
      }

      final List<String> sortedNodeIds = new ArrayList<>(nodeStates.keySet());

      final List<Entry<String, String>> sortedList = nodeStates.entrySet().stream().sorted(Map.Entry.comparingByKey())
            .collect(Collectors.toList());

      int cnt = 1;

      final StringBuilder sb = new StringBuilder();
      for(final Entry<String, String> e : sortedList) {
         sb.append(e.getValue());
         if(cnt % Math.sqrt(game.getGridnodes().size()) == 0) {
            sb.append("\n");
         }
         cnt++;
      }
      return sb.toString();
   }

}
