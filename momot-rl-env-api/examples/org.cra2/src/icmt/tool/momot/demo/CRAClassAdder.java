package icmt.tool.momot.demo;

import java.io.IOException;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;

import icmt.tool.momot.demo.architecture.ArchitectureFactory;
import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.Class;
import icmt.tool.momot.demo.architecture.ClassModel;
import icmt.tool.momot.demo.architecture.Feature;

public class CRAClassAdder {

   private static final String model = "problem/CRA_C.xmi";
   private static final String out = "problem/CRA_C_classed.xmi";

   public static void main(final String[] args) {
      // Initialize the EMF package
      ArchitecturePackage.eINSTANCE.eClass();

      final HenshinResourceSet resourceSet = new HenshinResourceSet();

      final Resource resource = resourceSet.getResource(URI.createURI(model), true);

      try {
         resource.load(null);
      } catch(final IOException e) {
         e.printStackTrace();
         return;
      }
      final ClassModel sm = (ClassModel) resource.getContents().get(0);

      final EClass classClass = (EClass) ArchitecturePackage.eINSTANCE.getEClassifier("Class");
      // Iterate through the model and modify the "load" attribute
      int i = 0;
      for(final Feature s : sm.getFeatures()) {
         final EObject newclassObj = ArchitectureFactory.eINSTANCE.create(classClass);
         final icmt.tool.momot.demo.architecture.Class newClass = (Class) newclassObj;
         newClass.setName("Class_" + i++);
         newClass.getEncapsulates().add(s);

         sm.getClasses().add(newClass);
      }

      // Save the modified model to a different location
      final URI newFileURI = URI.createFileURI(out);
      final Resource newResource = resourceSet.createResource(newFileURI);
      newResource.getContents().add(sm);

      try {
         newResource.save(null);
         System.out.println("Modification completed successfully. Model saved to: " + newFileURI);
      } catch(final IOException e) {
         e.printStackTrace();
      }
   }
}
