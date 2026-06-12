package co.gongzh.procbridge;

import org.jetbrains.annotations.Nullable;

public final class ServerException extends RuntimeException {

   private static final String UNKNOWN_SERVER_ERROR = "unknown server error";

   ServerException(@Nullable final String message) {
      super(message != null ? message : UNKNOWN_SERVER_ERROR);
   }

   public ServerException(final Throwable cause) {
      super(cause.getMessage(), cause);
   }

}
