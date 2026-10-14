package at.ac.tuwien.big.momot.learn;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EReference;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Encodes an Henshin {@link EGraph} as a compact JSON object of integer-typed
 * node/edge/attribute arrays, using a vocabulary supplied by the Python client.
 */
public class CompactGraphEncoder {

   private final Map<String, Integer> nodeTypeNames;
   private final Map<String, Integer> edgeTypeNames;
   private final String idAttribute;
   private final String[] numericAttributes;

   private final Map<EClass, Integer> classToTypeId = new IdentityHashMap<>();
   private final Map<EReference, Integer> refToTypeId = new IdentityHashMap<>();
   private final Map<EClass, EAttribute> classToIdAttr = new IdentityHashMap<>();
   private final Map<EClass, EAttribute[]> classToNumericAttrs = new IdentityHashMap<>();

   public CompactGraphEncoder(final Map<String, Integer> nodeTypeNames, final Map<String, Integer> edgeTypeNames,
         final String idAttribute, final List<String> numericAttributes) {
      this.nodeTypeNames = nodeTypeNames;
      this.edgeTypeNames = edgeTypeNames;
      this.idAttribute = idAttribute == null || idAttribute.isEmpty() ? null : idAttribute;
      this.numericAttributes = numericAttributes == null ? new String[0]
            : numericAttributes.toArray(new String[numericAttributes.size()]);
   }

   public static CompactGraphEncoder fromSpec(final JSONObject spec) {
      if(spec == null) {
         throw new IllegalArgumentException("observation spec must not be null");
      }
      final Map<String, Integer> nodeTypes = readIntNameToId(spec.getJSONObject("nodeTypes"));
      final Map<String, Integer> edgeTypes = readIntNameToId(spec.getJSONObject("edgeTypes"));
      final String idAttribute = spec.has("idAttribute") ? spec.optString("idAttribute") : null;
      final List<String> numericAttributeNames = new ArrayList<>();
      final JSONArray numericAttributeSpec = spec.optJSONArray("numericAttributes");
      if(numericAttributeSpec != null) {
         for(int i = 0; i < numericAttributeSpec.length(); i++) {
            numericAttributeNames.add(numericAttributeSpec.getString(i));
         }
      }
      return new CompactGraphEncoder(nodeTypes, edgeTypes, idAttribute, numericAttributeNames);
   }

   public JSONObject encode(final EGraph graph) {
      if(graph == null) {
         throw new IllegalArgumentException("graph must not be null");
      }

      final IdentityHashMap<EObject, Integer> objectToIndex = new IdentityHashMap<>();
      final List<EObject> keptObjects = new ArrayList<>();
      for(final EObject obj : graph) {
         if(obj == null) {
            continue;
         }
         final int typeId = typeId(obj.eClass());
         if(typeId < 0) {
            continue;
         }
         objectToIndex.put(obj, keptObjects.size());
         keptObjects.add(obj);
      }

      final int numNodes = keptObjects.size();
      final JSONArray nodeTypes = new JSONArray();
      final JSONArray nodeIds = new JSONArray();
      final JSONArray nodeAttributes = numericAttributes.length == 0 ? null : new JSONArray();

      for(final EObject obj : keptObjects) {
         nodeTypes.put(typeId(obj.eClass()));
         if(idAttribute != null) {
            final Object val = identifierValue(obj);
            nodeIds.put(val == null ? JSONObject.NULL : String.valueOf(val));
         }
         if(nodeAttributes != null) {
            nodeAttributes.put(numericAttributeRow(obj));
         }
      }

      final JSONArray edgeSources = new JSONArray();
      final JSONArray edgeTargets = new JSONArray();
      final JSONArray edgeTypes = new JSONArray();
      for(final EObject obj : keptObjects) {
         final int sourceIndex = objectToIndex.get(obj);
         for(final EReference ref : obj.eClass().getEAllReferences()) {
            if(ref.isDerived()) {
               continue;
            }
            final int edgeType = edgeTypeId(ref);
            if(edgeType < 0) {
               continue;
            }
            final Object val = obj.eGet(ref);
            if(val == null) {
               continue;
            }
            if(ref.isMany()) {
               for(final Object target : (Collection<?>) val) {
                  emitEdge(objectToIndex, edgeSources, edgeTargets, edgeTypes, sourceIndex, target, edgeType);
               }
            } else {
               emitEdge(objectToIndex, edgeSources, edgeTargets, edgeTypes, sourceIndex, val, edgeType);
            }
         }
      }

      final JSONObject observation = new JSONObject();
      observation.put("numNodes", numNodes);
      observation.put("nodeTypes", nodeTypes);
      if(idAttribute != null) {
         observation.put("nodeIds", nodeIds);
      }
      if(nodeAttributes != null) {
         observation.put("nodeAttributes", nodeAttributes);
      }
      observation.put("edgeSources", edgeSources);
      observation.put("edgeTargets", edgeTargets);
      observation.put("edgeTypes", edgeTypes);
      return observation;
   }

   private static void emitEdge(final IdentityHashMap<EObject, Integer> objectToIndex, final JSONArray edgeSources,
         final JSONArray edgeTargets, final JSONArray edgeTypes, final int sourceIndex, final Object target,
         final int edgeType) {
      if(!(target instanceof EObject)) {
         return;
      }
      final Integer targetIndex = objectToIndex.get(target);
      if(targetIndex == null) {
         return;
      }
      edgeSources.put(sourceIndex);
      edgeTargets.put(targetIndex.intValue());
      edgeTypes.put(edgeType);
   }

   private int typeId(final EClass cls) {
      final Integer cached = classToTypeId.get(cls);
      if(cached != null) {
         return cached.intValue();
      }
      final Integer byName = nodeTypeNames.get(cls.getName());
      final int id = byName == null ? -1 : byName.intValue();
      classToTypeId.put(cls, id);
      return id;
   }

   private int edgeTypeId(final EReference ref) {
      final Integer cached = refToTypeId.get(ref);
      if(cached != null) {
         return cached.intValue();
      }
      final Integer byName = edgeTypeNames.get(ref.getName());
      final int id = byName == null ? -1 : byName.intValue();
      refToTypeId.put(ref, id);
      return id;
   }

   private Object identifierValue(final EObject obj) {
      final EAttribute attr = idAttribute(obj.eClass());
      if(attr == null) {
         return null;
      }
      return obj.eGet(attr);
   }

   private EAttribute idAttribute(final EClass cls) {
      if(classToIdAttr.containsKey(cls)) {
         return classToIdAttr.get(cls);
      }
      EAttribute found = null;
      if(idAttribute != null) {
         for(final EAttribute attr : cls.getEAllAttributes()) {
            if(idAttribute.equals(attr.getName())) {
               found = attr;
               break;
            }
         }
      }
      classToIdAttr.put(cls, found);
      return found;
   }

   private JSONArray numericAttributeRow(final EObject obj) {
      final EAttribute[] attrs = numericAttributes(obj.eClass());
      final JSONArray row = new JSONArray();
      for(int i = 0; i < attrs.length; i++) {
         double value = 0.0;
         if(attrs[i] != null) {
            final Object raw = obj.eGet(attrs[i]);
            if(raw instanceof Number) {
               value = ((Number) raw).doubleValue();
            } else if(raw instanceof Boolean) {
               value = ((Boolean) raw).booleanValue() ? 1.0 : 0.0;
            }
         }
         row.put(value);
      }
      return row;
   }

   private EAttribute[] numericAttributes(final EClass cls) {
      EAttribute[] cached = classToNumericAttrs.get(cls);
      if(cached != null) {
         return cached;
      }
      cached = new EAttribute[numericAttributes.length];
      for(int i = 0; i < numericAttributes.length; i++) {
         final String name = numericAttributes[i];
         for(final EAttribute attr : cls.getEAllAttributes()) {
            if(name.equals(attr.getName())) {
               cached[i] = attr;
               break;
            }
         }
      }
      classToNumericAttrs.put(cls, cached);
      return cached;
   }

   private static Map<String, Integer> readIntNameToId(final JSONObject obj) {
      final Map<String, Integer> map = new HashMap<>();
      if(obj == null) {
         return map;
      }
      for(final String key : obj.keySet()) {
         map.put(key, obj.getInt(key));
      }
      return map;
   }
}
