package co.gongzh.procbridge;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.Map;
import java.util.concurrent.Executor;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import co.gongzh.procbridge.ClientException;
import co.gongzh.procbridge.ServerException;
import co.gongzh.procbridge.TimeoutException;

/**
 * @author Gong Zhang
 */
public class Client {

   public static final long FOREVER = 0;
   private final @NotNull String host;
   private final int port;
   private final long timeout;

   private final @Nullable Executor executor;

   public Client(@NotNull final String host, final int port) {
      this(host, port, FOREVER, null);
   }

   public Client(@NotNull final String host, final int port, final long timeout, @Nullable final Executor executor) {
      this.host = host;
      this.port = port;
      this.timeout = timeout;
      this.executor = executor;
   }

   @Nullable
   public Executor getExecutor() {
      return executor;
   }

   @NotNull
   public final String getHost() {
      return host;
   }

   public final int getPort() {
      return port;
   }

   public long getTimeout() {
      return timeout;
   }

   @Nullable
   public final Object request(@Nullable final String method, @Nullable final Object payload)
         throws ClientException, TimeoutException, ServerException {
      final StatusCode[] respStatusCode = { null };
      final Object[] respPayload = { null };
      final Throwable[] innerException = { null };

      try(final Socket socket = new Socket(host, port)) {
         final Runnable task = () -> {
            try(OutputStream os = socket.getOutputStream(); InputStream is = socket.getInputStream()) {

               Protocol.writeRequest(os, method, payload);
               final Map.Entry<StatusCode, Object> entry = Protocol.readResponse(is);
               respStatusCode[0] = entry.getKey();
               respPayload[0] = entry.getValue();

            } catch(final Exception ex) {
               innerException[0] = ex;
            }
         };

         if(timeout <= 0) {
            task.run();
         } else {
            final TimeoutExecutor guard = new TimeoutExecutor(timeout, executor);
            guard.execute(task);
         }
      } catch(final IOException ex) {
         throw new ClientException(ex);
      }

      if(innerException[0] != null) {
         throw new RuntimeException(innerException[0]);
      }

      if(respStatusCode[0] != StatusCode.GOOD_RESPONSE) {
         throw new ServerException((String) respPayload[0]);
      }

      return respPayload[0];
   }

}
