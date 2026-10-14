package co.gongzh.procbridge;

import static co.gongzh.procbridge.ProtocolException.INCOMPATIBLE_VERSION;
import static co.gongzh.procbridge.ProtocolException.INCOMPLETE_DATA;
import static co.gongzh.procbridge.ProtocolException.INVALID_BODY;
import static co.gongzh.procbridge.ProtocolException.INVALID_STATUS_CODE;
import static co.gongzh.procbridge.ProtocolException.UNRECOGNIZED_PROTOCOL;

import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.AbstractMap;
import java.util.Map;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.json.JSONObject;

final class Protocol {

   private static final byte[] FLAG = { 'p', 'b' };

   private static @NotNull Map.Entry<StatusCode, JSONObject> read(@NotNull final InputStream stream)
         throws IOException, ProtocolException {
      int b;

      // 1. FLAG
      b = stream.read();

      if(b == -1) {
         throw new EOFException();
      }
      if(b != FLAG[0]) {
         throw new ProtocolException(UNRECOGNIZED_PROTOCOL);
      }
      b = stream.read();
      if(b == -1 || b != FLAG[1]) {
         throw new ProtocolException(UNRECOGNIZED_PROTOCOL);
      }

      // 2. VERSION
      b = stream.read();
      if(b == -1) {
         throw new ProtocolException(INCOMPLETE_DATA);
      }
      if(b != Versions.CURRENT[0]) {
         throw new ProtocolException(INCOMPATIBLE_VERSION);
      }
      b = stream.read();
      if(b == -1) {
         throw new ProtocolException(INCOMPLETE_DATA);
      }
      if(b != Versions.CURRENT[1]) {
         throw new ProtocolException(INCOMPATIBLE_VERSION);
      }

      // 3. STATUS CODE
      b = stream.read();
      if(b == -1) {
         throw new ProtocolException(INCOMPLETE_DATA);
      }
      final StatusCode statusCode = StatusCode.fromRawValue(b);
      if(statusCode == null) {
         throw new ProtocolException(INVALID_STATUS_CODE);
      }

      // 4. RESERVED BYTES (2 bytes)
      b = stream.read();
      if(b == -1) {
         throw new ProtocolException(INCOMPLETE_DATA);
      }
      b = stream.read();
      if(b == -1) {
         throw new ProtocolException(INCOMPLETE_DATA);
      }

      // 5. LENGTH (little endian)
      int bodyLen;
      b = stream.read();
      if(b == -1) {
         throw new ProtocolException(INCOMPLETE_DATA);
      }
      bodyLen = b;
      b = stream.read();
      if(b == -1) {
         throw new ProtocolException(INCOMPLETE_DATA);
      }
      bodyLen |= b << 8;
      b = stream.read();
      if(b == -1) {
         throw new ProtocolException(INCOMPLETE_DATA);
      }
      bodyLen |= b << 16;
      b = stream.read();
      if(b == -1) {
         throw new ProtocolException(INCOMPLETE_DATA);
      }
      bodyLen |= b << 24;

      // 6. JSON OBJECT
      final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
      int readCount;
      int restCount = bodyLen;
      byte[] buf = new byte[Math.min(bodyLen, 1024 * 1024)];
      while((readCount = stream.read(buf, 0, Math.min(buf.length, restCount))) != -1) {
         buffer.write(buf, 0, readCount);
         restCount -= readCount;
         if(restCount == 0) {
            break;
         }
      }

      if(buffer.size() != bodyLen) {
         throw new ProtocolException(INCOMPLETE_DATA);
      }

      buffer.flush();
      buf = buffer.toByteArray();

      try {
         final String jsonText = new String(buf, StandardCharsets.UTF_8);
         final JSONObject body = new JSONObject(jsonText);
         return new AbstractMap.SimpleEntry<>(statusCode, body);
      } catch(final Exception ex) {
         throw new ProtocolException(INVALID_BODY);
      }
   }

   static @NotNull Map.Entry<String, Object> readRequest(@NotNull final InputStream stream)
         throws IOException, ProtocolException {
      final Map.Entry<StatusCode, JSONObject> entry = read(stream);
      final StatusCode statusCode = entry.getKey();
      final JSONObject body = entry.getValue();
      if(statusCode != StatusCode.REQUEST) {
         throw new ProtocolException(INVALID_STATUS_CODE);
      }
      final String method = body.optString(Keys.METHOD);
      final Object payload = body.opt(Keys.PAYLOAD);
      return new AbstractMap.SimpleEntry<>(method, payload);
   }

   static @NotNull Map.Entry<StatusCode, Object> readResponse(@NotNull final InputStream stream)
         throws IOException, ProtocolException {
      final Map.Entry<StatusCode, JSONObject> entry = read(stream);
      final StatusCode statusCode = entry.getKey();
      final JSONObject body = entry.getValue();
      if(statusCode == StatusCode.GOOD_RESPONSE) {
         return new AbstractMap.SimpleEntry<>(StatusCode.GOOD_RESPONSE, body.opt(Keys.PAYLOAD));
      } else if(statusCode == StatusCode.BAD_RESPONSE) {
         return new AbstractMap.SimpleEntry<>(StatusCode.BAD_RESPONSE, body.optString(Keys.MESSAGE));
      } else {
         throw new ProtocolException(INVALID_STATUS_CODE);
      }
   }

   private static void write(@NotNull final OutputStream stream, @NotNull final StatusCode statusCode,
         @NotNull final JSONObject body) throws IOException {
      // 1. FLAG 'p', 'b'
      stream.write(FLAG);

      // 2. VERSION
      stream.write(Versions.CURRENT);

      // 3. STATUS CODE
      stream.write(statusCode.rawValue);

      // 4. RESERVED BYTES (2 bytes)
      stream.write(0);
      stream.write(0);

      // make json object
      final byte[] buf = body.toString().getBytes(StandardCharsets.UTF_8);

      // 5. LENGTH (4-byte, little endian)
      final int len = buf.length;
      final int b0 = len & 0xff;
      final int b1 = (len & 0xff00) >> 8;
      final int b2 = (len & 0xff0000) >> 16;
      final int b3 = (len & 0xff000000) >> 24;
      stream.write(b0);
      stream.write(b1);
      stream.write(b2);
      stream.write(b3);

      // 6. JSON OBJECT
      stream.write(buf);

      stream.flush();
   }

   static void writeBadResponse(@NotNull final OutputStream stream, @Nullable final String message) throws IOException {
      final JSONObject body = new JSONObject();
      if(message != null) {
         body.put(Keys.MESSAGE, message);
      }
      write(stream, StatusCode.BAD_RESPONSE, body);
   }

   static void writeGoodResponse(@NotNull final OutputStream stream, @Nullable final Object payload)
         throws IOException {
      final JSONObject body = new JSONObject();
      if(payload != null) {
         body.put(Keys.PAYLOAD, payload);
      }
      write(stream, StatusCode.GOOD_RESPONSE, body);
   }

   static void writeRequest(@NotNull final OutputStream stream, @Nullable final String method,
         @Nullable final Object payload) throws IOException {
      final JSONObject body = new JSONObject();
      if(method != null) {
         body.put(Keys.METHOD, method);
      }
      if(payload != null) {
         body.put(Keys.PAYLOAD, payload);
      }
      write(stream, StatusCode.REQUEST, body);
   }

   private Protocol() {
   }

}
