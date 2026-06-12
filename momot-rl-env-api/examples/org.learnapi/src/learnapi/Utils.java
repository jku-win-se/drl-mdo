package learnapi;

import at.ac.tuwien.big.momot.examples.stack.stack.StackModel;
import at.ac.tuwien.big.momot.examples.stack.stack.StackPackage;

import java.io.IOException;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.json.JSONObject;

import SprintPlanning.Plan;
import SprintPlanning.SprintPlanningPackage;
import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.ClassModel;
import learnapi.ConditionUtils.LimitType;

public class Utils {

   public static int getCRATransformationLength(final String path) {
      ArchitecturePackage.eINSTANCE.eClass();
      final Resource resource = loadModel(path);

      ClassModel sm = null;
      for(final Object obj : resource.getContents()) {
         if(obj instanceof ClassModel) {
            sm = (ClassModel) obj;
         }
      }
      return sm.getFeatures().size() * 2;
   }

   public static JSONObject getScrumConditions(final String path) {
      SprintPlanningPackage.eINSTANCE.eClass();
      final Resource resource = loadModel(path);

      Plan p = null;
      for(final Object obj : resource.getContents()) {
         if(obj instanceof Plan) {
            p = (Plan) obj;
         }
      }

      final int items = p.getBacklog().getWorkitems().size();

      return new ConditionUtils().addCondition(0, "elements", "Sprint", LimitType.MAX, p.getMaxSprints())
            .addCondition(2, "elements", "Sprint", LimitType.MIN, p.getMinSprints())
            .addCondition(4, "elements", "Sprint", LimitType.MAX, p.getMaxSprints())
            .addCondition(1, "relations", "isPlannedFor", LimitType.MIN, items - 1)
            .addCondition(2, "relations", "isPlannedFor", LimitType.MIN, items - 1)
            .addCondition(4, "relations", "isPlannedFor", LimitType.MIN, items - 1).build();
   }

   public static Plan getScrumModel(final String path) {
      SprintPlanningPackage.eINSTANCE.eClass();

      final Resource resource = loadModel(path);

      // Retrieve StackModel
      Plan sm = null;
      for(final Object obj : resource.getContents()) {
         if(obj instanceof Plan) {
            sm = (Plan) obj;
            return sm;
         }
      }
      return sm;
   }

   public static int getScrumTransformationLength(final String path) {
      SprintPlanningPackage.eINSTANCE.eClass();
      final Resource resource = loadModel(path);

      Plan sm = null;
      for(final Object obj : resource.getContents()) {
         if(obj instanceof Plan) {
            sm = (Plan) obj;
         }
      }
      return (int) Math.floor(sm.getBacklog().getWorkitems().size() * 2.3);
   }

   public static int getSLBTransformationLength(final String path) {
      StackPackage.eINSTANCE.eClass();
      final Resource resource = loadModel(path);

      StackModel sm = null;
      for(final Object obj : resource.getContents()) {
         if(obj instanceof StackModel) {
            sm = (StackModel) obj;
         }
      }
      return (int) Math.floor(sm.getStacks().size() * 2.5);
   }

   private static Resource loadModel(final String path) {
      Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

      final ResourceSet r = new ResourceSetImpl();

      final Resource resource = r.getResource(URI.createFileURI(path), true);

      // Ensure model is loaded
      try {
         resource.load(null);
         return resource;
      } catch(final IOException e) {
         e.printStackTrace();
         return null;
      }
   }
}
