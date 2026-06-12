package at.ac.tuwien.big.momot.examples.stack;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;

import org.eclipse.emf.ecore.resource.Resource;

public class LearnAPIUtils {
   static OutputStream transformXMIToString(final Resource resource) {
      // Create a StringWriter to store the XMI as a string
      final ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
      // Save the resource to the StringWriter
      try {
         resource.save(byteStream, null);
         byteStream.flush();

      } catch(final IOException e) {
         e.printStackTrace();
      }
      // Get the resulting string
      return byteStream;
   }

   static void writeFileAndConsole(final PrintWriter pw, final String s) {
      pw.println(s);
      System.out.println(s);
   }

}
