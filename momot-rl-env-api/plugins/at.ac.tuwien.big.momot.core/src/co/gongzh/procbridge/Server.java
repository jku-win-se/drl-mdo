package co.gongzh.procbridge;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

/**
 * @author Gong Zhang
 */
public class Server {

   final class Connection implements Runnable {

      private final Socket socket;
      private final IDelegate delegate;

      Connection(final Socket socket, final IDelegate delegate) {
         this.socket = socket;
         this.delegate = delegate;
      }

      @Override
      public void run() {
         try {
            socket.setTcpNoDelay(true);
            try(InputStream is = new BufferedInputStream(socket.getInputStream(), 65536);
                  OutputStream os = new BufferedOutputStream(socket.getOutputStream(), 65536)) {
               while(true) {
                  final Map.Entry<String, Object> req;
                  try {
                     req = Protocol.readRequest(is);
                  } catch(final EOFException eof) {
                     break;
                  }

                  Object result = null;
                  Exception exception = null;
                  try {
                     result = delegate.handleRequest(req.getKey(), req.getValue());
                  } catch(final Exception ex) {
                     exception = ex;
                  }

                  if(exception != null) {
                     Protocol.writeBadResponse(os, exception.getMessage());
                  } else {
                     Protocol.writeGoodResponse(os, result);
                  }
               }
            }
         } catch(final EOFException | SocketException ex) {
            // client closed the connection
         } catch(final Exception ex) {
            if(logger != null) {
               ex.printStackTrace(logger);
            }
         } finally {
            try {
               socket.close();
            } catch(final IOException ignored) {
            }
         }
      }

   }

   private final int port;

   private final @NotNull IDelegate delegate;
   private ExecutorService executor;
   private ServerSocket serverSocket;
   private Thread listenerThread;

   private boolean started;

   private @Nullable PrintStream logger;

   public Server(final int port, @NotNull final IDelegate delegate) {
      this.port = port;
      this.delegate = delegate;

      this.started = false;
      this.executor = null;
      this.serverSocket = null;
      this.logger = System.err;
      this.listenerThread = null;

   }

   public Thread getListenerThread() {
      return this.listenerThread;
   }

   @Nullable
   public PrintStream getLogger() {
      return logger;
   }

   public final int getPort() {
      return port;
   }

   public final synchronized boolean isStarted() {
      return started;
   }

   public void setLogger(@Nullable final PrintStream logger) {
      this.logger = logger;
   }

   public synchronized void start() throws IOException {
      if(started) {
         throw new IllegalStateException("server already started");
      }

      serverSocket = new ServerSocket(this.port);
      executor = Executors.newCachedThreadPool();
      started = true;

      listenerThread = new Thread(() -> {
         while(!Thread.currentThread().isInterrupted()) {
            try {
               final Socket socket = serverSocket.accept();
               final Connection conn = new Connection(socket, delegate);
               synchronized(Server.this) {
                  if(!started) {
                     break;
                  }
                  executor.execute(conn);
               }
            } catch(final IOException e) {
               // Stop if server socket was closed
               if(started && logger != null) {
                  logger.println("Server stopped accepting connections: " + e.getMessage());
               }
               break;
            }
         }
      });
      listenerThread.start();
   }

   public synchronized void stop() {
      if(!started) {
         throw new IllegalStateException("server not started");
      }

      started = false;

      // Close the server socket (will cause accept() to throw)
      try {
         if(serverSocket != null && !serverSocket.isClosed()) {
            serverSocket.close();
         }
      } catch(final IOException e) {
         if(logger != null) {
            logger.println("Error closing server socket: " + e.getMessage());
         }
      }

      // Stop the listener thread
      if(listenerThread != null && listenerThread.isAlive()) {
         listenerThread.interrupt();
         try {
            listenerThread.join(1000); // optional: wait for up to 1s
         } catch(final InterruptedException ignored) {
         }
      }
      listenerThread = null;

      // Shut down the executor service
      if(executor != null) {
         executor.shutdownNow(); // Or graceful: executor.shutdown(); awaitTermination(...)
         executor = null;
      }

      serverSocket = null;
   }

}
