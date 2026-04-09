package icmt.tool.momot.demo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;

import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.ClassModel;
import icmt.tool.momot.demo.architecture.Feature;
import icmt.tool.momot.demo.architecture.Method;

public class CRAClassMutator {
   private static final String model = "problem/nn/Cart_Item_classed.xmi";
   private static final String out = "problem/nn/";
   private static int genModels = 50;

   public static void main(final String[] args) {
      // Initialize the EMF package
      ArchitecturePackage.eINSTANCE.eClass();

      for(int i = 0; i < genModels; i++) {
         final HenshinResourceSet resourceSet = new HenshinResourceSet();
         final Resource resource = resourceSet.getResource(URI.createURI(model), true);
         final Random r = new Random();

         try {
            resource.load(null);
         } catch(final IOException e) {
            e.printStackTrace();
            return;
         }
         final ClassModel sm = (ClassModel) resource.getContents().get(0);

         final EClass classClass = (EClass) ArchitecturePackage.eINSTANCE.getEClassifier("Class");
         // Iterate through the model and modify the "load" attribute
         // int removeN = 1;
         // if(r.nextFloat() > .8) {
         // removeN = 2;
         // }

         // for(int j = 0; j < removeN; j++) {
         // final Feature eo = sm.getFeatures().get(r.nextInt(sm.getFeatures().size()));
         // EcoreUtil.delete(eo);
         // ;
         //
         // }
         for(final Feature s : sm.getFeatures()) {
            if(s instanceof Method && r.nextFloat() > .3) {
               s.setIsEncapsulatedBy(sm.getClasses().get(r.nextInt(sm.getClasses().size())));
               final Method m = (Method) s;
               if(!m.getFunctionalDependency().isEmpty()) {
                  m.getFunctionalDependency().remove(r.nextInt(m.getFunctionalDependency().size()));
               }
               final Method mTarget = (Method) sm.getFeatures().stream().filter(f -> f instanceof Method && f != m)
                     .findAny().get();
               m.getFunctionalDependency().add(mTarget);

            }
         }
         final List<EObject> removeClassesList = new ArrayList<>();
         for(final icmt.tool.momot.demo.architecture.Class c : sm.getClasses()) {
            if(c.getEncapsulates().isEmpty()) {
               removeClassesList.add(c);
            }
         }
         sm.getClasses().removeAll(removeClassesList);
         // Save the modified model to a different location
         final URI newFileURI = URI.createFileURI(out + "CRA_A_" + i + ".xmi");
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
}
