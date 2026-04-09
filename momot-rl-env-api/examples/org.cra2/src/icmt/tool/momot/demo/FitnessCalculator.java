package icmt.tool.momot.demo;

import java.util.ArrayList;
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

   public static double calculateCohesion(final ClassModel model) {
      double cohesionRatio = 0.0;
      for(final Class clazz : model.getClasses()) {
         if(getMethodsClass(clazz).size() == 0) {
            cohesionRatio += 0.0;
         } else if(getMethodsClass(clazz).size() == 1) { // Here, the second part of the addition is still 0
            if(getAttributesClass(clazz).size() == 0) { // and now, also the first part is 0
               cohesionRatio += 0.0;
            } else { // now, the first part is not 0
               cohesionRatio += mai(clazz, clazz)
                     / (double) (getMethodsClass(clazz).size() * getAttributesClass(clazz).size());
            }
         } else { // Here, we have more than one method in the clazz
            if(getAttributesClass(clazz).size() == 0) { // Now, the first part of the addition will be 0
               cohesionRatio += mmi(clazz, clazz)
                     / (double) (getMethodsClass(clazz).size() * (getMethodsClass(clazz).size() - 1));
            } else { // Here, we have more than 0 attributes and more than 1 method, so we use the whole formula
               cohesionRatio += mai(clazz, clazz)
                     / (double) (getMethodsClass(clazz).size() * getAttributesClass(clazz).size())
                     + mmi(clazz, clazz)
                           / (double) (getMethodsClass(clazz).size() * (getMethodsClass(clazz).size() - 1));
            }
         }
      }
      return cohesionRatio;
   }

   /**
    * Cache for non-variable model counts.
    *
    * Assumption per requirement: the model instance may change across evaluations,
    * but counts (#classes/#features/#attributes/#methods) do not.
    */
   // private static final class ModelStatsCache {
   // int nrClasses;
   // int nrFeatures;
   // int nrAttributes;
   // int nrMethods;
   // }
   //
   // /**
   // * Multiple caches indexed by (nrFeatures, nrClasses).
   // */
   // private static final Map<Long, ModelStatsCache> STATS_BY_INDEX = new HashMap<>();
   //
   // private static long cacheIndex(final int nrFeatures, final int nrClasses) {
   // // Pack two non-negative ints into a long key.
   // return (long) nrFeatures << 32 ^ nrClasses & 0xffffffffL;
   // }

   // static double calculateCohesion(final ClassModel model) {
   // double cohesionRatio = 0.0;
   // for(final Class clazz : model.getClasses()) {
   // if(getMethodsClass(clazz).size() == 0) {
   // cohesionRatio += 0.0;
   // } else if(getMethodsClass(clazz).size() == 1) { // Here, the second part of the addition is still 0
   // if(getAttributesClass(clazz).size() == 0) { // and now, also the first part is 0
   // cohesionRatio += 0.0;
   // } else { // now, the first part is not 0
   // cohesionRatio += mai(clazz, clazz)
   // / (double) (getMethodsClass(clazz).size() * getAttributesClass(clazz).size());
   // }
   // } else { // Here, we have more than one method in the clazz
   // if(getAttributesClass(clazz).size() == 0) { // Now, the first part of the addition will be 0
   // cohesionRatio += mmi(clazz, clazz)
   // / (double) (getMethodsClass(clazz).size() * (getMethodsClass(clazz).size() - 1));
   // } else { // Here, we have more than 0 attributes and more than 1 method, so we use the whole formula
   // cohesionRatio += mai(clazz, clazz)
   // / (double) (getMethodsClass(clazz).size() * getAttributesClass(clazz).size())
   // + mmi(clazz, clazz)
   // / (double) (getMethodsClass(clazz).size() * (getMethodsClass(clazz).size() - 1));
   // }
   // }
   // }
   // return cohesionRatio;
   // }
   //
   // static double calculateCoupling(final Class classSource, final ClassModel model) {
   // double couplingRatio = 0;
   // for(final Class classTarget : model.getClasses()) {
   // if(classSource != classTarget) {
   // if(getMethodsClass(classSource).size() == 0) {
   // couplingRatio += 0.0;
   // } else { // From here, |M(clsi)|>0
   // if(getMethodsClass(classTarget).size() <= 1) { // The second part of the addition is 0
   // if(getAttributesClass(classTarget).size() == 0) { // Now, also the first part of the addition is 0
   // couplingRatio += 0.0;
   // } else { // Now, the first part of the addition is not 0
   // couplingRatio += mai(classSource, classTarget)
   // / (double) (getMethodsClass(classSource).size() * getAttributesClass(classTarget).size());
   // }
   // } else { // Now, the second part of the addition is not 0
   // if(getAttributesClass(classTarget).size() == 0) {
   // couplingRatio += mmi(classSource, classTarget)
   // / (double) (getMethodsClass(classSource).size() * (getMethodsClass(classTarget).size() - 1));
   // } else { // Now, non of the parts is 0
   // couplingRatio += mai(classSource, classTarget)
   // / (double) (getMethodsClass(classSource).size() * getAttributesClass(classTarget).size())
   // + mmi(classSource, classTarget) / (double) (getMethodsClass(classSource).size()
   // * (getMethodsClass(classTarget).size() - 1));
   // }
   // }
   // }
   // }
   // }
   // return couplingRatio;
   // }
   //
   // static double calculateCoupling(final ClassModel model) {
   // double couplingRatio = 0;
   // for(final Class clazz : model.getClasses()) {
   // couplingRatio += calculateCoupling(clazz, model);
   // }
   // return couplingRatio;
   // }
   // public static double calculateCohesion(final ClassModel model) {
   // final ModelStatsCache stats = getOrInitCache(model);
   //
   // // First, calculate TOTAL actual dependencies in the entire model
   // int totalActualDependencies = 0;
   // for(final Class clazz : model.getClasses()) {
   // totalActualDependencies += mai(clazz, clazz) + mmi(clazz, clazz);
   // for(final Class other : model.getClasses()) {
   // if(clazz != other) {
   // totalActualDependencies += mai(clazz, other) + mmi(clazz, other);
   // }
   // }
   // }
   // // Avoid double-counting: divide by 2 if dependencies are symmetric
   // // (based on your implementation, they seem directional, so no division needed)
   //
   // if(totalActualDependencies == 0) {
   // return 0.0; // No dependencies at all
   // }
   //
   // // Now calculate intra-class dependencies
   // int totalIntraClassDependencies = 0;
   // for(final Class clazz : model.getClasses()) {
   // totalIntraClassDependencies += mai(clazz, clazz) + mmi(clazz, clazz);
   // }
   //
   // // Cohesion = proportion of all dependencies that are WITHIN classes
   // return (double) totalIntraClassDependencies / totalActualDependencies;
   // }
   //
   // public static double calculateCoupling(final ClassModel model) {
   // final ModelStatsCache stats = getOrInitCache(model);
   //
   // // Calculate TOTAL actual dependencies in the entire model
   // int totalActualDependencies = 0;
   // for(final Class clazz : model.getClasses()) {
   // totalActualDependencies += mai(clazz, clazz) + mmi(clazz, clazz);
   // for(final Class other : model.getClasses()) {
   // if(clazz != other) {
   // totalActualDependencies += mai(clazz, other) + mmi(clazz, other);
   // }
   // }
   // }
   //
   // if(totalActualDependencies == 0) {
   // return 0.0; // No dependencies at all
   // }
   //
   // // Calculate inter-class dependencies
   // int totalInterClassDependencies = 0;
   // for(final Class clazz : model.getClasses()) {
   // for(final Class other : model.getClasses()) {
   // if(clazz != other) {
   // totalInterClassDependencies += mai(clazz, other) + mmi(clazz, other);
   // }
   // }
   // }
   //
   // // Coupling = proportion of all dependencies that are BETWEEN classes
   // return (double) totalInterClassDependencies / totalActualDependencies;
   // }
   public static double calculateCohesionNorm(final ClassModel model) {

      double cohesionRatio = 0.0;
      for(final Class clazz : model.getClasses()) {
         final int nrMethods = getMethodsClass(clazz).size();
         final int nrAttributes = getAttributesClass(clazz).size();
         final int m = nrMethods + nrAttributes; // number of features in current class

         // Maximum number of (undirected) intra-class feature dependencies.
         // If m < 2, __denom__ becomes 0 and the cohesion contribution is defined as 0.
         final double denom = m < 2 ? 0.0 : m * (m - 1) / 2.0;

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

         cohesionRatio += denom == 0.0 ? 0.0 : intraDependencies / denom;
      }

      return cohesionRatio / model.getClasses().size();
   }

   static double calculateCoupling(final Class classSource, final ClassModel model) {
      double couplingRatio = 0;
      for(final Class classTarget : model.getClasses()) {
         if(classSource != classTarget) {
            if(getMethodsClass(classSource).size() == 0) {
               couplingRatio += 0.0;
            } else { // From here, |M(clsi)|>0
               if(getMethodsClass(classTarget).size() <= 1) { // The second part of the addition is 0
                  if(getAttributesClass(classTarget).size() == 0) { // Now, also the first part of the addition is 0
                     couplingRatio += 0.0;
                  } else { // Now, the first part of the addition is not 0
                     couplingRatio += mai(classSource, classTarget)
                           / (double) (getMethodsClass(classSource).size() * getAttributesClass(classTarget).size());
                  }
               } else { // Now, the second part of the addition is not 0
                  if(getAttributesClass(classTarget).size() == 0) {
                     couplingRatio += mmi(classSource, classTarget)
                           / (double) (getMethodsClass(classSource).size() * (getMethodsClass(classTarget).size() - 1));
                  } else { // Now, non of the parts is 0
                     couplingRatio += mai(classSource, classTarget)
                           / (double) (getMethodsClass(classSource).size() * getAttributesClass(classTarget).size())
                           + mmi(classSource, classTarget) / (double) (getMethodsClass(classSource).size()
                                 * (getMethodsClass(classTarget).size() - 1));
                  }
               }
            }
         }
      }
      return couplingRatio;
   }

   // public static double calculateCohesion(final ClassModel model) {
   //
   // // First, calculate TOTAL actual dependencies in the entire model
   // int totalActualDependencies = 0;
   // for(final Class clazz : model.getClasses()) {
   // totalActualDependencies += mai(clazz, clazz) + mmi(clazz, clazz);
   // for(final Class other : model.getClasses()) {
   // if(clazz != other) {
   // totalActualDependencies += mai(clazz, other) + mmi(clazz, other);
   // }
   // }
   // }
   // // Avoid double-counting: divide by 2 if dependencies are symmetric
   // // (based on your implementation, they seem directional, so no division needed)
   //
   // if(totalActualDependencies == 0) {
   // return 0.0; // No dependencies at all
   // }
   //
   // // Now calculate intra-class dependencies
   // int totalIntraClassDependencies = 0;
   // for(final Class clazz : model.getClasses()) {
   // totalIntraClassDependencies += mai(clazz, clazz) + mmi(clazz, clazz);
   // }
   //
   // // Cohesion = proportion of all dependencies that are WITHIN classes
   // return (double) totalIntraClassDependencies / totalActualDependencies;
   // }
   //
   // public static double calculateCoupling(final ClassModel model) {
   //
   // // Calculate TOTAL actual dependencies in the entire model
   // int totalActualDependencies = 0;
   // for(final Class clazz : model.getClasses()) {
   // totalActualDependencies += mai(clazz, clazz) + mmi(clazz, clazz);
   // for(final Class other : model.getClasses()) {
   // if(clazz != other) {
   // totalActualDependencies += mai(clazz, other) + mmi(clazz, other);
   // }
   // }
   // }
   //
   // if(totalActualDependencies == 0) {
   // return 0.0; // No dependencies at all
   // }
   //
   // // Calculate inter-class dependencies
   // int totalInterClassDependencies = 0;
   // for(final Class clazz : model.getClasses()) {
   // for(final Class other : model.getClasses()) {
   // if(clazz != other) {
   // totalInterClassDependencies += mai(clazz, other) + mmi(clazz, other);
   // }
   // }
   // }
   //
   // // Coupling = proportion of all dependencies that are BETWEEN classes
   // return (double) totalInterClassDependencies / totalActualDependencies;
   // }

   public static double calculateCoupling(final ClassModel model) {
      double couplingRatio = 0;
      for(final Class clazz : model.getClasses()) {
         couplingRatio += calculateCoupling(clazz, model);
      }
      return couplingRatio;
   }

   static double calculateCouplingNorm(final Class classSource, final ClassModel model) {

      final int m = classSource.getEncapsulates().size();
      // number of features not in current class
      final int n = model.getFeatures().size() - m;
      final double denom = m == 0 || n <= 0 ? 0.0 : (double) m * (double) n;

      double interDependencies = 0.0;
      for(final Class classTarget : model.getClasses()) {
         if(classSource != classTarget) {
            if(getMethodsClass(classSource).size() == 0) {
               interDependencies += 0.0;
            } else { // From here, |M(__clsi__)|>0
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

      return denom == 0.0 ? 0.0 : interDependencies / denom;
   }

   public static double calculateCouplingNorm(final ClassModel model) {

      double couplingRatio = 0;
      for(final Class clazz : model.getClasses()) {
         couplingRatio += calculateCouplingNorm(clazz, model);
      }

      return couplingRatio / model.getClasses().size();
   }

   public static double calculateCRAIndex(final ClassModel model) {
      return calculateCohesion(model) - calculateCoupling(model);
   }

   // c
   public static double calculateCRAIndexNorm(final ClassModel model) {
      return calculateCohesionNorm(model) - calculateCouplingNorm(model);
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

   // private static ModelStatsCache getOrInitCache(final ClassModel model) {
   // final int nrClasses = model.getClasses().size();
   // final int nrFeatures = model.getFeatures().size();
   // final long idx = cacheIndex(nrFeatures, nrClasses);
   //
   // ModelStatsCache stats = STATS_BY_INDEX.get(idx);
   // if(stats != null) {
   // return stats;
   // }
   //
   // stats = new ModelStatsCache();
   // stats.nrClasses = nrClasses;
   // stats.nrFeatures = nrFeatures;
   //
   // int methods = 0;
   // int attributes = 0;
   // for(final Feature f : model.getFeatures()) {
   // if(f instanceof Method) {
   // methods++;
   // } else if(f instanceof Attribute) {
   // attributes++;
   // }
   // }
   // stats.nrMethods = methods;
   // stats.nrAttributes = attributes;
   //
   // STATS_BY_INDEX.put(idx, stats);
   // return stats;
   // }

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
      final double cohesion = calculateCohesionNorm(model);
      final double coupling = calculateCouplingNorm(model);
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

   private static int sumAllPairs(final ClassModel model) {
      int sum = 0;
      for(final Class ic : model.getClasses()) {
         for(final Class oc : model.getClasses()) {
            if(ic != oc) {
               sum += mmi(ic, oc) + mai(ic, oc);
            }
         }
      }
      return sum;
   }

   private FitnessCalculator() {
   }
}
