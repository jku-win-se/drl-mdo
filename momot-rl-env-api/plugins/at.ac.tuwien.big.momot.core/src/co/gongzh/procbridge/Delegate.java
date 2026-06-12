package co.gongzh.procbridge;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.json.JSONArray;

/**
 * @author Gong Zhang
 */
public abstract class Delegate implements IDelegate {

   @NotNull
   private final Map<String, Method> handlers;

   protected Delegate() {
      this.handlers = new HashMap<>();
      for(final Method m : this.getClass().getDeclaredMethods()) {
         if(m.getAnnotation(Handler.class) != null) {
            final String key = m.getName();
            if(handlers.containsKey(key)) {
               throw new UnsupportedOperationException("duplicate handler name: " + key);
            }
            m.setAccessible(true);
            handlers.put(key, m);
         }
      }
   }

   @Override
   public final @Nullable Object handleRequest(@Nullable final String method, @Nullable final Object payload) {
      willHandleRequest(method, payload);

      final Method m = handlers.get(method);
      if(m == null) {
         return handleUnknownRequest(method, payload);
      }

      Object result = null;
      try {
         final int pcnt = m.getParameterCount();
         if(pcnt == 0) {
            result = m.invoke(this);
         } else if(pcnt == 1) {
            result = m.invoke(this, payload);
         } else {
            // unpack
            if(!(payload instanceof JSONArray)) {
               throw new ServerException("payload must be an array");
            }
            final JSONArray arr = (JSONArray) payload;
            if(arr.length() != pcnt) {
               throw new ServerException(String.format("method needs %d elements", pcnt));
            }
            result = m.invoke(this, arr);
         }
      } catch(final InvocationTargetException e) {
         throw new ServerException(e.getTargetException());
      } catch(final IllegalAccessException e) {
         throw new ServerException(e);
      }

      return result;
   }

   protected @Nullable Object handleUnknownRequest(@Nullable final String method, @Nullable final Object payload) {
      throw new ServerException("unknown method: " + method);
   }

   protected void willHandleRequest(@Nullable final String method, @Nullable final Object payload) {
   }

}
