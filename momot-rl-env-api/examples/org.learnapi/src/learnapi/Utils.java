package learnapi;

import at.ac.tuwien.big.momot.examples.stack.stack.StackModel;
import at.ac.tuwien.big.momot.examples.stack.stack.StackPackage;

import java.io.IOException;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.json.JSONObject;

import SprintPlanning.Plan;
import SprintPlanning.SprintPlanningPackage;
import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.ClassModel;

public class Utils {

   public static int getCRATransformationLength(final String path) {
      ArchitecturePackage.eINSTANCE.eClass();
      return UseCase.calculateSteps("cra", requireRoot(loadModel(path), ClassModel.class));
   }

   public static JSONObject getScrumConditions(final String path) {
      SprintPlanningPackage.eINSTANCE.eClass();
      return UseCase.calculateConditions("scrum-uc", requireRoot(loadModel(path), Plan.class));
   }

   public static Plan getScrumModel(final String path) {
      SprintPlanningPackage.eINSTANCE.eClass();
      return requireRoot(loadModel(path), Plan.class);
   }

   public static int getScrumTransformationLength(final String path) {
      SprintPlanningPackage.eINSTANCE.eClass();
      return UseCase.calculateSteps("scrum", requireRoot(loadModel(path), Plan.class));
   }

   public static int getSLBTransformationLength(final String path) {
      StackPackage.eINSTANCE.eClass();
      return UseCase.calculateSteps("slb", requireRoot(loadModel(path), StackModel.class));
   }

   private static Resource loadModel(final String path) {
      Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

      final ResourceSet r = new ResourceSetImpl();

      final Resource resource = r.getResource(URI.createFileURI(path), true);

      try {
         resource.load(null);
         return resource;
      } catch(final IOException e) {
         e.printStackTrace();
         return null;
      }
   }

   private static <T extends EObject> T requireRoot(final Resource resource, final Class<T> type) {
      if(resource == null) {
         throw new IllegalStateException("Model resource is null");
      }
      for(final Object obj : resource.getContents()) {
         if(type.isInstance(obj)) {
            return type.cast(obj);
         }
      }
      throw new IllegalStateException("No " + type.getSimpleName() + " found in resource");
   }
}
