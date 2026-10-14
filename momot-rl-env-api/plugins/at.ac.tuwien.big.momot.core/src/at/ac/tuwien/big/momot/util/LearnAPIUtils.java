package at.ac.tuwien.big.momot.util;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.xmi.XMLResource;

public class LearnAPIUtils {

   private static final Logger LOGGER = Logger.getLogger(LearnAPIUtils.class.getName());
   private static final Object CONSOLE_LOCK = new Object();

   /**
    * Transforms an EMF Resource to XMI format and returns it as a UTF-8 string.
    * This method is thread-safe as it only uses local variables and the Resource
    * save operation is synchronized on the resource itself.
    *
    * @param resource the EMF resource to transform
    * @return UTF-8 XMI representation of the resource
    */
   public static String transformXMIToString(final Resource resource) {
      if(resource == null) {
         throw new IllegalArgumentException("Resource cannot be null");
      }

      final ByteArrayOutputStream byteStream = new ByteArrayOutputStream();

      synchronized(resource) {
         try {
            final Map<String, Object> options = new HashMap<>();
            options.put(XMLResource.OPTION_ENCODING, "UTF-8");
            resource.save(byteStream, options);
            byteStream.flush();
         } catch(final IOException e) {
            LOGGER.log(Level.SEVERE, "Failed to save resource to XMI format", e);
            throw new RuntimeException("Failed to transform resource to XMI", e);
         }
      }

      return new String(byteStream.toByteArray(), StandardCharsets.UTF_8);
   }

   /**
    * Writes a string to both a PrintWriter and the console in a thread-safe manner.
    * The console output is synchronized to prevent interleaved output in multi-threaded
    * environments.
    *
    * @param pw the PrintWriter to write to
    * @param s the string to write
    */
   public static void writeFileAndConsole(final PrintWriter pw, final String s) {
      if(pw == null || s == null) {
         throw new IllegalArgumentException("PrintWriter and string cannot be null");
      }

      synchronized(pw) {
         pw.println(s);
         pw.flush();
      }

      synchronized(CONSOLE_LOCK) {
         System.out.println(s);
         System.out.flush();
      }
   }

}
