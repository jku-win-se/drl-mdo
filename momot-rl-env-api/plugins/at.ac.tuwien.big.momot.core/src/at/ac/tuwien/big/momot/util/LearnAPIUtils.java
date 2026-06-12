package at.ac.tuwien.big.momot.util;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.eclipse.emf.ecore.resource.Resource;

public class LearnAPIUtils {
   
   private static final Logger LOGGER = Logger.getLogger(LearnAPIUtils.class.getName());
   private static final Object CONSOLE_LOCK = new Object();
   
   /**
    * Transforms an EMF Resource to XMI format and returns it as an OutputStream.
    * This method is thread-safe as it only uses local variables and the Resource
    * save operation is synchronized on the resource itself.
    * 
    * @param resource the EMF resource to transform
    * @return ByteArrayOutputStream containing the XMI representation
    */
   public static OutputStream transformXMIToString(final Resource resource) {
      if (resource == null) {
         throw new IllegalArgumentException("Resource cannot be null");
      }
      
      final ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
      
      // Synchronize on the resource to ensure thread-safe access to EMF Resource
      synchronized (resource) {
         try {
            resource.save(byteStream, null);
            byteStream.flush();
         } catch (final IOException e) {
            // Use proper logging instead of printStackTrace to avoid interleaved output
            LOGGER.log(Level.SEVERE, "Failed to save resource to XMI format", e);
            // Re-throw as runtime exception to maintain method contract
            throw new RuntimeException("Failed to transform resource to XMI", e);
         }
      }
      
      return byteStream;
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
      if (pw == null || s == null) {
         throw new IllegalArgumentException("PrintWriter and string cannot be null");
      }
      
      // File writing - PrintWriter is typically thread-safe for individual operations
      // but we synchronize on the PrintWriter to ensure atomic write+flush
      synchronized (pw) {
         pw.println(s);
         pw.flush();
      }
      
      // Console writing - use a static lock to ensure atomic console output
      synchronized (CONSOLE_LOCK) {
         System.out.println(s);
         System.out.flush();
      }
   }

}
