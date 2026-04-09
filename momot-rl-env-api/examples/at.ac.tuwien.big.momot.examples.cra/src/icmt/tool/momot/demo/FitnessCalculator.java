package icmt.tool.momot.demo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.Attribute;
import icmt.tool.momot.demo.architecture.Class;
import icmt.tool.momot.demo.architecture.ClassModel;
import icmt.tool.momot.demo.architecture.Feature;
import icmt.tool.momot.demo.architecture.Method;

public final class FitnessCalculator {

   /**
    * Cache for non-variable model counts.
    *
    * Multiple models can exist; the correct cache entry is selected via an index
    * derived from the number of classes and features.
    */
   private static final class ModelStatsCache {
      final int nrClasses;
      final int nrFeatures;
      final int nrAttributes;
      final int nrMethods;

      private ModelStatsCache(final int nrClasses, final int nrFeatures, final int nrAttributes, final int nrMethods) {
         this.nrClasses = nrClasses;
         this.nrFeatures = nrFeatures;
         this.nrAttributes = nrAttributes;
         this.nrMethods = nrMethods;
      }
   }

   /**
    * Keyed by an index derived from (nrClasses, nrFeatures).
    */
   private static final Map<Long, ModelStatsCache> STATS_CACHE = new HashMap<>();

   private static long statsIndex(final int nrClasses, final int nrFeatures) {
      // Pack two non-negative ints into a long (stable across runs)
      return (((long) nrClasses) << 32) ^ (nrFeatures & 0xffffffffL);
   }

   private static ModelStatsCache getStats(final ClassModel model) {
      final int nrClasses = model.getClasses().size();
      final int nrFeatures = model.getFeatures().size();
      final long idx = statsIndex(nrClasses, nrFeatures);

      ModelStatsCache stats = STATS_CACHE.get(idx);
      if(stats != null) {
         return stats;
      }

      int methods = 0;
      int attributes = 0;
      for(final Feature f : model.getFeatures()) {
         if(f instanceof Method) {
            methods++;
         } else if(f instanceof Attribute) {
            attributes++;
         }
      }

      stats = new ModelStatsCache(nrClasses, nrFeatures, attributes, methods);
      STATS_CACHE.put(idx, stats);
      return stats;
   }

   public static double calculateCohesion(final ClassModel model) {
      final ModelStatsCache stats = getStats(model);

      double cohesionRatio = 0.0;
      for(final Class clazz : model.getClasses()) {
         final int nrMethods = getMethodsClass(clazz).size();
         final int nrAttributes = getAttributesClass(clazz).size();
         final int m = nrMethods + nrAttributes; // number of features in current class

         // Maximum number of (undirected) intra-class feature dependencies.
         // If m < 2, denom becomes 0 and the cohesion contribution is defined as 0.
         final double denom = (m < 2) ? 0.0 : (m * (m - 1)) / 2.0;

         double intraDependencies = 0.0;
         if(nrMethods == 0) {
            intraDependencies += 0.0;
         } else if(nrMethods == 1) {
            if(nrAttributes == 0) {
               intraDependencies += 0.0;
            } else {
               intraDependencies += mai(clazz, clazz);
            }
         } else {
            if(nrAttributes == 0) {
               intraDependencies += mmi(clazz, clazz);
            } else {
               intraDependencies += mai(clazz, clazz) + mmi(clazz, clazz);
            }
         }

         cohesionRatio += (denom == 0.0) ? 0.0 : (intraDependencies / denom);
      }

      return (stats.nrClasses == 0) ? 0.0 : (cohesionRatio / stats.nrClasses);
   }

   static double calculateCoupling(final Class classSource, final ClassModel model) {
      final ModelStatsCache stats = getStats(model);

      final int m = classSource.getEncapsulates().size();
      // number of features not in current class
      final int n = stats.nrFeatures - m;
      final double denom = (m == 0 || n <= 0) ? 0.0 : (double) m * (double) n;

      double interDependencies = 0.0;
      for(final Class classTarget : model.getClasses()) {
         if(classSource != classTarget) {
            if(getMethodsClass(classSource).size() == 0) {
               interDependencies += 0.0;
            } else { // From here, |M(clsi)|>0
               if(getMethodsClass(classTarget).size() <= 1) { // The second part of the addition is 0
                  if(getAttributesClass(classTarget).size() == 0) { // Now, also the first part of the addition is 0
                     interDependencies += 0.0;
                  } else { // Now, the first part of the addition is not 0
                     interDependencies += mai(classSource, classTarget);
                  }
               } else { // Now, the second part of the addition is not 0
                  if(getAttributesClass(classTarget).size() == 0) {
                     interDependencies += mmi(classSource, classTarget);
                  } else { // Now, non of the parts is 0
                     interDependencies += mai(classSource, classTarget) + mmi(classSource, classTarget);
                  }
               }
            }
         }
      }

      return (denom == 0.0) ? 0.0 : (interDependencies / denom);
   }

   public static double calculateCoupling(final ClassModel model) {
      final ModelStatsCache stats = getStats(model);

      double couplingRatio = 0;
      for(final Class clazz : model.getClasses()) {
         couplingRatio += calculateCoupling(clazz, model);
      }
      return (stats.nrClasses == 0) ? 0.0 : (couplingRatio / stats.nrClasses);
   }

   public static double calculateCRAIndex(final ClassModel model) {
      return calculateCohesion(model) - calculateCoupling(model);
   }

   static boolean checkAllClassesDifferentNames(final ClassModel model) {
      boolean violated = false;
      final List<String> classesNames = new ArrayList<>();
      for(final Class clazz : model.getClasses()) {
         if(classesNames.contains(clazz.getName())) {
            System.err.println(
                  "Constraint violated! Unique Names for Classes. The name " + clazz.getName() + " is repeated");
            violated = true;
         } else {
            classesNames.add(clazz.getName());
         }
      }
      if(!violated) {
         System.out.println("All classes have different names: ok");
      }
      return violated;
   }

   static boolean checkAllFeaturesEncapsulated(final ClassModel model) {
      boolean violated = false;
      for(final Feature feature : model.getFeatures()) {
         if(feature.getIsEncapsulatedBy() == null) {
            System.err.println("Constraint violated! Feature " + feature.getName() + " is not encapsulated in a class");
            violated = true;
         }
      }
      if(!violated) {
         System.out.println("All features are encapsulated in a class: ok");
      }
      return violated;
   }

   public static void evaluateModel(final ClassModel model) {
      if(model == null) {
         System.err.println("No correct model loaded... abort.");
         return;
      }

      printGeneralInfo(model);
      printCorrectnessInfo(model);
      printOptimalityInfo(model);
   }

   public static void evaluateModel(final String model) {
      registerPackage();
      evaluateModel(loadModel(model));
   }

   static List<Attribute> getAttributesClass(final Class clazz) {
      final List<Attribute> attributes = new ArrayList<>();
      for(final Feature feature : clazz.getEncapsulates()) {
         if(feature instanceof Attribute) {
            attributes.add((Attribute) feature);
         }
      }
      return attributes;
   }

   static List<Method> getMethodsClass(final Class clazz) {
      final List<Method> methods = new ArrayList<>();
      for(final Feature feature : clazz.getEncapsulates()) {
         if(feature instanceof Method) {
            methods.add((Method) feature);
         }
      }
      return methods;
   }

   public static ClassModel loadModel(final String model) {
      final ResourceSet resSet = new ResourceSetImpl();
      final Resource resource = resSet.getResource(URI.createURI(model), true);
      if(resource == null) {
         System.err.println("Model can not be loaded!");
         return null;
      }
      final EObject root = resource.getContents().get(0);
      if(!(root instanceof ClassModel)) {
         System.err.println("Model is not a ClassModel");
         return null;
      }
      return (ClassModel) resource.getContents().get(0);
   }

   static int mai(final Class classSource, final Class classTarget) {
      int mai = 0;
      for(final Method method : getMethodsClass(classSource)) {
         for(final Attribute attribute : getAttributesClass(classTarget)) {
            if(method.getDataDependency().contains(attribute)) {
               mai++;
            }
         }
      }
      return mai;
   }

   public static void main(final String[] args) {
      if(args.length == 0) {
         System.err.println("Please provide the model as parameter.");
         System.err.println("Example: java -jar CRAIndexCalculator.jar ToEvaluate.xmi");
         return;
      }
      evaluateModel(args[0]);
   }

   static int mmi(final Class classSource, final Class classTarget) {
      int mmi = 0;
      for(final Method methodSource : getMethodsClass(classSource)) {
         for(final Method methodTarget : getMethodsClass(classTarget)) {
            if(methodSource.getFunctionalDependency().contains(methodTarget)) {
               mmi++;
            }
         }
      }
      return mmi;
   }

   private static void printCorrectnessInfo(final ClassModel model) {
      System.out.println("------------------------------------------");
      System.out.println("Correctness");
      System.out.println("------------------------------------------");
      final boolean classNames = checkAllClassesDifferentNames(model);
      final boolean featuresEncapsulated = checkAllFeaturesEncapsulated(model);
      if(!classNames && !featuresEncapsulated) {
         System.out.println("Correctness: ok");
      } else {
         System.out.println("Correctness: Violations found.");
      }
      System.out.println("------------------------------------------\n");
   }

   public static void printGeneralInfo(final ClassModel model) {
      final int nrClasses = model.getClasses().size();
      int nrMethods = 0;
      int nrAttributes = 0;
      for(final Feature f : model.getFeatures()) {
         if(f instanceof Method) {
            nrMethods++;
         }
         if(f instanceof Attribute) {
            nrAttributes++;
         }
      }
      System.out.println("------------------------------------------");
      System.out.println("General Info");
      System.out.println("------------------------------------------");
      System.out.println("|Classes|    = " + nrClasses);
      System.out.println("|Methods|    = " + nrMethods);
      System.out.println("|Attributes| = " + nrAttributes);
      System.out.println("------------------------------------------\n");
   }

   public static void printOptimalityInfo(final ClassModel model) {
      final double cohesion = calculateCohesion(model);
      final double coupling = calculateCoupling(model);
      final double craindex = cohesion - coupling;
      System.out.println("------------------------------------------");
      System.out.println("Optimality");
      System.out.println("------------------------------------------");
      System.out.println("The aggregated cohesion ratio is: " + cohesion);
      System.out.println("The aggregated coupling ratio is: " + coupling);
      System.out.println("This makes a CRA-Index of: " + craindex);
      System.out.println("------------------------------------------\n");
   }

   public static void registerPackage() {
      ArchitecturePackage.eINSTANCE.eClass();

      final Resource.Factory.Registry reg = Resource.Factory.Registry.INSTANCE;
      final Map<String, Object> m = reg.getExtensionToFactoryMap();
      m.put("xmi", new XMIResourceFactoryImpl());
   }

   private FitnessCalculator() {
   }
}
