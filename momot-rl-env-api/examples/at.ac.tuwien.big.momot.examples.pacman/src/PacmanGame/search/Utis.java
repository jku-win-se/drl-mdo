package PacmanGame.search;

import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
import org.eclipse.emf.henshin.statespace.Model;
import org.eclipse.emf.henshin.statespace.impl.ModelImpl;

import PacmanGame.GridNode;
import PacmanGame.PacmanGamePackage;
import PacmanGame.PositionableEntity;
import PacmanGame.impl.FoodImpl;
import PacmanGame.impl.GameImpl;
import PacmanGame.impl.GhostImpl;
import PacmanGame.impl.PacmanImpl;

public class Utis {

   protected final static String[] modules = new String[] { "transformation/play_game.henshin" };

   protected static final String INITIAL_MODEL = "Modelset/initial_models/4x4_f3_g2_0.xmi";

   protected static EGraph createInputGraph(final String initialGraph, final ModuleManager moduleManager) {
      final EGraph graph = moduleManager.loadGraph(initialGraph);
      return graph;
   }

   protected static ModuleManager createModuleManager() {
      final ModuleManager manager = new ModuleManager();
      for(final String module : modules) {
         manager.addModule(URI.createFileURI(new File(module).getPath().toString()).toString());
      }
      return manager;
   }

   public static void main(final String[] args) {
      PacmanGamePackage.eINSTANCE.getClass();

      final HenshinResourceSet henshinResourceSet = new HenshinResourceSet();
      final Resource modelResource = henshinResourceSet.getResource(INITIAL_MODEL);
      final Model initialModel = new ModelImpl(modelResource);
      printModel(initialModel.getEGraph());
   }

   public static void printModel(final EGraph graph) {
      final GameImpl game = MomotUtil.getRoot(graph, GameImpl.class);
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

      for(final Entry<String, String> e : sortedList) {
         System.out.print(e.getValue());
         if(cnt % Math.sqrt(game.getGridnodes().size()) == 0) {
            System.out.println();
         }
         cnt++;
      }
      // for(final PositionableEntity entity : game.getEntites()) {
      // if(entity instanceof PacmanImpl) {
      // fieldToChar.put(entity.getOn().getId(), "P");
      //
      // } else if(entity instanceof GhostImpl) {
      // fieldToChar.put(entity.getOn().getId(), "G");
      //
      // } else if(entity instanceof FoodImpl) {
      // fieldToChar.put(entity.getOn().getId(), "F");
      //
      // }
      // }
      // final List<GridNode> nodes = game.getGridnodes();
      // int i = 0;
      // for(final GridNode n : nodes) {
      // if(fieldToChar.containsKey(n.getId())) {
      // System.out.print(fieldToChar.get(n.getId()) + " ");
      // } else {
      // System.out.print("- ");
      // }
      // if(++i % Math.sqrt(nodes.size()) == 0) {
      // System.out.print("\n");
      //
      // }
      // }
   }

}
