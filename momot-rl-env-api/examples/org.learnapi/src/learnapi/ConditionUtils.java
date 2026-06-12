package learnapi;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.json.JSONObject;

public class ConditionUtils {

   public enum LimitType {
      MAX("max"), MIN("min");

      private final String key;

      LimitType(final String key) {
         this.key = key;
      }

      public String getKey() {

         return key;
      }
   }

   // Example usage
   public static void main(final String[] args) {
      final ConditionUtils builder = new ConditionUtils();
      builder.addCondition(3, "elements", "Sprint", LimitType.MAX, 15)
            .addCondition(3, "elements", "Speed", LimitType.MIN, 5)
            .addCondition(4, "metrics", "Accuracy", LimitType.MAX, 95);

      final JSONObject result = builder.build();
      System.out.println(result.toString(2));
   }

   public static <T> List<T> repeat(final List<T> list, final int times) {
      return IntStream.range(0, times).boxed().flatMap(i -> list.stream()).collect(Collectors.toList());
   }

   private final JSONObject conditions;

   public ConditionUtils() {
      this.conditions = new JSONObject();
   }

   /**
    * Adds a condition entry of the form:
    * { index: { type: { objectName: { minmax: value } } } }
    * Example:
    * addCondition(3, "elements", "Sprint", LimitType.MAX, 15)
    * produces:
    * { "conditions": { "3": { "elements": { "Sprint": { "max": 15 } } } } }
    */
   public ConditionUtils addCondition(final int index, final String type, final String objectName,
         final LimitType minmax, final Integer value) {
      final String indexKey = String.valueOf(index);

      // Ensure index object exists
      JSONObject indexObject = conditions.optJSONObject(indexKey);
      if(indexObject == null) {
         indexObject = new JSONObject();
         conditions.put(indexKey, indexObject);
      }

      // Ensure type object exists
      JSONObject typeObject = indexObject.optJSONObject(type);
      if(typeObject == null) {
         typeObject = new JSONObject();
         indexObject.put(type, typeObject);
      }

      // Ensure objectName object exists
      JSONObject objectEntry = typeObject.optJSONObject(objectName);
      if(objectEntry == null) {
         objectEntry = new JSONObject();
         typeObject.put(objectName, objectEntry);
      }

      // Add the limit entry ("max": value or "min": value)
      objectEntry.put(minmax.getKey(), value);

      return this; // Allow chaining
   }

   /**
    * Wraps the accumulated structure under "conditions"
    * and returns the final JSONObject.
    */
   public JSONObject build() {
      final JSONObject root = new JSONObject();
      root.put("conditions", conditions);
      return root;
   }
}
