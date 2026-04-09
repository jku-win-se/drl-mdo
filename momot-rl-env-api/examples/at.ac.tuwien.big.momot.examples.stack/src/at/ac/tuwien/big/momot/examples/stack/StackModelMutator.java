package at.ac.tuwien.big.momot.examples.stack;

import at.ac.tuwien.big.momot.examples.stack.stack.Stack;
import at.ac.tuwien.big.momot.examples.stack.stack.StackModel;
import at.ac.tuwien.big.momot.examples.stack.stack.StackPackage;

import java.io.IOException;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;

public class StackModelMutator {
   // Utility method to get a random number in a specified range
   private static int getRandomNumberInRange(final int min, final int max) {
      return min + (int) (Math.random() * (max - min + 1));
   }

   public static void main(final String[] args) {
      // Initialize the EMF package
      StackPackage.eINSTANCE.eClass();
      int genNr = 1;
      // Load the model
      for(int i = 0; i < 10; i++) {
         final HenshinResourceSet resourceSet = new HenshinResourceSet();

         final Resource resource = resourceSet.getResource(URI.createURI("model/model_five_stacks.xmi"), true);

         try {
            resource.load(null);
         } catch(final IOException e) {
            e.printStackTrace();
            return;
         }
         final StackModel sm = (StackModel) resource.getContents().get(0);
         // Iterate through the model and modify the "load" attribute
         for(final Stack s : sm.getStacks()) {
            // Modify the "load" attribute here
            int newLoadValue = s.getLoad() + getRandomNumberInRange(-5, 5);
            // Ensure load value is not less than 1
            newLoadValue = Math.max(newLoadValue, 1);
            s.setLoad(newLoadValue);

         }

         // Save the modified model to a different location
         final URI newFileURI = URI
               .createFileURI("modelset/input/5_mutations/" + String.format("%02d", genNr++) + ".xmi");
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
