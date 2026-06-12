package at.ac.tuwien.big.momot.examples.stack;

import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.examples.stack.stack.Stack;
import at.ac.tuwien.big.momot.examples.stack.stack.StackModel;
import at.ac.tuwien.big.momot.examples.stack.stack.StackPackage;
import at.ac.tuwien.big.momot.search.engine.MomotEngine;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.interpreter.RuleApplication;
import org.eclipse.emf.henshin.interpreter.impl.EGraphImpl;
import org.eclipse.emf.henshin.interpreter.impl.ProfilingApplicationMonitor;

public class StackCreator {
   public static final String EMPTY_MODEL = "empty_model.xmi";

   public static final StackCreator INSTANCE = new StackCreator();

   private static final int INITIAL_VALUE = 0;

   private static final int STEP_SIZE = 1;

   private static int STACK_INDEX = 0;

   private static final String BASE_NAME = "Stack_";

   private static String currentName() {
      return previousName(0);
   }

   private static String firstName() {
      return BASE_NAME + (INITIAL_VALUE + STEP_SIZE);
   }

   public static void main(final String[] args) {
      StackCreator.INSTANCE.createStacksGraphDirect(80, "model_80_stacks.xmi");
   }

   private static String newName() {
      STACK_INDEX += STEP_SIZE;
      return currentName();
   }

   protected static String nextName() {
      return nextName(1);
   }

   private static String nextName(final int steps) {
      return BASE_NAME + (STACK_INDEX + steps * STEP_SIZE);
   }

   private static String previousName() {
      return previousName(1);
   }

   private static String previousName(final int steps) {
      return BASE_NAME + (STACK_INDEX - steps * STEP_SIZE);
   }

   private final ModuleManager henshin = new ModuleManager("model/");

   public StackCreator() {
      henshin.addModule("stack_full.henshin");
   }

   public EGraph createStacksGraph(final int nrStacks, final Integer[] loads, final String targetResource) {
      if(nrStacks != loads.length) {
         throw new IllegalArgumentException("Must provide load for all stacks.");
      }
      final EGraph graph = new EGraphImpl(henshin.getResourceSet().getResource(EMPTY_MODEL));

      final MomotEngine engine = new MomotEngine(false);
      final RuleApplication createStackApp = henshin.createRuleApplication(engine, "stack_full::Stack::createStack",
            graph);

      final RuleApplication connectStacksApp = henshin.createRuleApplication(engine, "stack_full::Stack::connectStacks",
            graph);

      final ProfilingApplicationMonitor monitor = new ProfilingApplicationMonitor();
      for(int i = 0; i < nrStacks; i++) {
         createStackApp.setParameterValue("stackId", newName());
         createStackApp.setParameterValue("stackLoad", loads[i]);
         createStackApp.execute(monitor);

         if(i > 0) {
            connectStacksApp.setParameterValue("left", previousName());
            connectStacksApp.setParameterValue("right", currentName());
            connectStacksApp.execute(monitor);
         }
      }
      connectStacksApp.setParameterValue("left", currentName());
      connectStacksApp.setParameterValue("right", firstName());
      connectStacksApp.execute(monitor);

      // monitor.printStats();

      henshin.saveGraph(graph, targetResource);

      return graph;
   }

   public EGraph createStacksGraph(final int nrStacks, final String targetResource) {
      final Integer[] loads = new Integer[nrStacks];
      for(int i = 0; i < nrStacks; i++) {
         loads[i] = new Random().nextInt(100);
      }
      return createStacksGraph(nrStacks, loads, targetResource);
   }

   public EGraph createStacksGraph(final Integer[] loads, final String targetResource) {
      return createStacksGraph(loads.length, loads, targetResource);
   }

   public EGraph createStacksGraphDirect(final int nrStacks, final String targetResource) {

      // Create resource and EGraph
      final Resource resource = henshin.getResourceSet().createResource(URI.createFileURI(targetResource));
      final EGraph graph = new EGraphImpl(resource);

      // Create a list to hold stacks

      final StackModel sm = (StackModel) EcoreUtil.create(StackPackage.eINSTANCE.getStackModel());

      // Create stacks
      for(int i = 0; i < nrStacks; i++) {
         final Stack stack = (Stack) EcoreUtil.create(StackPackage.eINSTANCE.getStack());
         stack.setId("Stack_" + i);
         stack.setLoad(new Random().nextInt(100));

         sm.getStacks().add(stack);
      }

      final List<Stack> stacks = sm.getStacks();
      // Connect stacks in a ring
      for(int i = 0; i < stacks.size(); i++) {
         final Stack left = i == 0 ? stacks.get(stacks.size() - 1) : stacks.get(i - 1);
         final Stack right = i == stacks.size() - 1 ? stacks.get(0) : stacks.get(i + 1);
         stacks.get(i).setLeft(left);
         stacks.get(i).setRight(right);
      }

      resource.getContents().add(sm);

      // Save resource
      try {
         resource.save(Collections.emptyMap());
      } catch(final IOException e) {
         e.printStackTrace();
      }

      return graph;
   }
}
