package learnapi;

import at.ac.tuwien.big.momot.examples.stack.stack.Stack;
import at.ac.tuwien.big.momot.examples.stack.stack.StackFactory;
import at.ac.tuwien.big.momot.examples.stack.stack.StackModel;
import at.ac.tuwien.big.momot.examples.stack.stack.StackPackage;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Stream;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.eclipse.emf.common.util.TreeIterator;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EReference;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.XMLResource;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.json.JSONObject;

import SprintPlanning.Backlog;
import SprintPlanning.Plan;
import SprintPlanning.SprintPlanningFactory;
import SprintPlanning.SprintPlanningPackage;
import SprintPlanning.Stakeholder;
import SprintPlanning.WorkItem;
import icmt.tool.momot.demo.architecture.ArchitectureFactory;
import icmt.tool.momot.demo.architecture.ArchitecturePackage;
import icmt.tool.momot.demo.architecture.Attribute;
import icmt.tool.momot.demo.architecture.ClassModel;
import icmt.tool.momot.demo.architecture.Feature;
import icmt.tool.momot.demo.architecture.Method;

/**
 * Generates a dataset of case-study models from their metamodels.
 *
 * The same arguments give the same dataset. Sizes are log-uniform, stratified for validation and test.
 * Non-size properties are drawn per model from ranges measured on the existing datasets, widened to
 * cover the benchmarks. Duplicates and copies of excluded models are rejected by a content hash.
 *
 * Output: train-N/ for each --train size (smaller sets are symlinks into the largest, so they are
 * contained in larger ones), val/, test/, and manifest.csv. Models are named order-nSIZE.xmi.
 */
public class ModelGenerator {

   private static final int MAX_ATTEMPTS = 100;

   // Model characteristics: edit by hand and rebuild (server-executables/build.sh). Ranges are
   // {min, max}, inclusive. The drawn values of each model are stored in manifest.csv.

   // Size per case, used when --sizes is not given: from 0.9 x the smallest to 1.1 x the largest
   // benchmark, rounded down.
   private static final int[] CRA_FEATURES = { 16, 176 };
   private static final int[] SLB_STACKS = { 9, 110 };
   private static final int[] SP_ITEMS = { 37, 136 };

   // Drawn uniformly per model; measured on the existing datasets, widened to cover the benchmarks.
   private static final double[] CRA_ATTRIBUTE_SHARE = { 0.30, 0.65 }; // Share of features that are attributes
   private static final double[] CRA_DEPENDENCY_DENSITY = { 0.05, 0.26 }; // Chance of each possible dependency
   private static final double[] SLB_MEAN_LOAD = { 5.8, 54.5 };
   private static final int[] SP_STAKEHOLDERS = { 4, 6 };
   private static final double[] SP_STAKEHOLDER_SKEW = { 0.0, 1.5 }; // 0: items spread evenly over stakeholders
   private static final double[] SP_MIN_SPRINTS_PER_ITEM = { 0.10, 0.18 };
   private static final double[] SP_MAX_SPRINTS_PER_ITEM = { 0.17, 0.26 };

   private final String caseId;
   private final Random random;
   private final Set<String> seen = new HashSet<>();
   private final List<String> manifest = new ArrayList<>();

   private ModelGenerator(final String caseId, final long seed) {
      this.caseId = caseId;
      this.random = new Random(seed);
   }

   public static void main(final String[] args) throws IOException, ParseException {
      final Options options = new Options();
      options.addOption(null, "case", true, "cra, slb or scrum");
      options.addOption(null, "sizes", true, "Size range MIN-MAX (features, stacks or work items); default: the case's range below");
      options.addOption(null, "train", true, "Comma-separated training-set sizes, e.g. 20,40,80");
      options.addOption(null, "val", true, "Number of validation models");
      options.addOption(null, "test", true, "Number of test models");
      options.addOption(null, "seed", true, "Generator seed");
      options.addOption(null, "exclude", true, "Folder of models the dataset must not contain; repeatable");
      options.addOption(null, "out", true, "Output folder");
      final CommandLine cmd = new DefaultParser().parse(options, args);

      final int[] defaultSizes = defaultSizes(cmd.getOptionValue("case"));
      final String[] range = cmd.getOptionValue("sizes", defaultSizes[0] + "-" + defaultSizes[1]).split("-");
      final int minSize = Integer.parseInt(range[0]);
      final int maxSize = Integer.parseInt(range[range.length - 1]);
      final int[] trainSizes = Arrays.stream(cmd.getOptionValue("train").split(",")).mapToInt(Integer::parseInt)
            .sorted().toArray();
      final int largest = trainSizes[trainSizes.length - 1];

      final ModelGenerator generator = new ModelGenerator(cmd.getOptionValue("case"),
            Long.parseLong(cmd.getOptionValue("seed")));
      if(cmd.hasOption("exclude")) {
         for(final String folder : cmd.getOptionValues("exclude")) {
            generator.exclude(new File(folder));
         }
      }

      final Path out = Path.of(cmd.getOptionValue("out"));
      final Path temporary = Path.of(out + ".tmp");
      deleteRecursively(temporary);
      Files.createDirectories(temporary);

      final String largestFolder = "train-" + largest;
      generator.generateSplit("train", temporary.resolve(largestFolder), largestFolder, largest, minSize, maxSize,
            false);
      generator.generateSplit("val", temporary.resolve("val"), "val", Integer.parseInt(cmd.getOptionValue("val", "0")),
            minSize, maxSize, false);
      generator.generateSplit("test", temporary.resolve("test"), "test",
            Integer.parseInt(cmd.getOptionValue("test", "0")), minSize, maxSize, false);

      // File names sort in generation order, so the first n are contained in any larger set.
      final File[] order = temporary.resolve(largestFolder).toFile().listFiles();
      Arrays.sort(order);
      for(final int size : trainSizes) {
         if(size == largest) {
            continue;
         }
         final Path folder = temporary.resolve("train-" + size);
         Files.createDirectories(folder);
         for(int i = 0; i < size; i++) {
            Files.createSymbolicLink(folder.resolve(order[i].getName()),
                  Path.of("..", largestFolder, order[i].getName()));
         }
      }

      try(PrintWriter writer = new PrintWriter(temporary.resolve("manifest.csv").toFile(), StandardCharsets.UTF_8)) {
         writer.println("id,split,size,seed,properties,hash");
         generator.manifest.forEach(writer::println);
      }

      Files.createDirectories(out.getParent());
      Files.move(temporary, out, StandardCopyOption.ATOMIC_MOVE);
      System.out.println("Generated " + generator.manifest.size() + " models in " + out);
   }

   private static void deleteRecursively(final Path path) throws IOException {
      if(!Files.exists(path)) {
         return;
      }
      try(Stream<Path> paths = Files.walk(path)) {
         paths.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
   }

   private void exclude(final File folder) {
      final File[] files = folder.listFiles((dir, name) -> name.endsWith(".xmi"));
      if(files == null) {
         throw new IllegalArgumentException("Cannot read excluded models in " + folder);
      }
      for(final File file : files) {
         seen.add(canonicalHash(newResourceSet().getResource(URI.createFileURI(file.getAbsolutePath()), true)));
      }
   }

   private void generateSplit(final String split, final Path folder, final String folderName, final int count,
         final int minSize, final int maxSize, final boolean stratified) throws IOException {
      if(count == 0) {
         return;
      }
      Files.createDirectories(folder);
      final List<Integer> sizes = sampleSizes(count, minSize, maxSize, stratified);
      for(int i = 0; i < count; i++) {
         final int size = sizes.get(i);
         final String id = String.format("%04d-n%d", i, size);
         final Path path = folder.resolve(id + ".xmi");
         generateNewModel(split, folderName, id, size, path);
      }
   }

   /** Log-uniform sizes; stratified: one draw per equal-width log interval, in random order. */
   private List<Integer> sampleSizes(final int count, final int minSize, final int maxSize, final boolean stratified) {
      final double low = Math.log(minSize);
      final double high = Math.log(maxSize + 1.0);
      final List<Integer> sizes = new ArrayList<>();
      for(int i = 0; i < count; i++) {
         final double draw = stratified ? low + (i + random.nextDouble()) * (high - low) / count
               : low + random.nextDouble() * (high - low);
         sizes.add(Math.min((int) Math.floor(Math.exp(draw)), maxSize));
      }
      if(stratified) {
         Collections.shuffle(sizes, random);
      }
      return sizes;
   }

   private void generateNewModel(final String split, final String folderName, final String id, final int size,
         final Path path) throws IOException {
      for(int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
         final long modelSeed = random.nextInt(Integer.MAX_VALUE);
         final JSONObject properties = sampleProperties(size);
         final EObject root = createModel(size, properties, new Random(modelSeed));
         final Resource resource = newResourceSet().createResource(URI.createFileURI(path.toAbsolutePath().toString()));
         resource.getContents().add(root);
         final String hash = canonicalHash(resource);
         if(seen.add(hash)) {
            final Map<String, Object> saveOptions = new HashMap<>();
            saveOptions.put(XMLResource.OPTION_ENCODING, "ASCII");
            resource.save(saveOptions);
            manifest.add(String.join(",", id, split, Integer.toString(size), Long.toString(modelSeed),
                  "\"" + properties.toString().replace("\"", "\"\"") + "\"", hash));
            return;
         }
      }
      throw new IllegalStateException(
            "Could not generate a new " + caseId + " model of size " + size + " in " + MAX_ATTEMPTS + " attempts");
   }

   private static int[] defaultSizes(final String caseId) {
      switch(caseId) {
         case "cra":
            return CRA_FEATURES;
         case "slb":
            return SLB_STACKS;
         case "scrum":
         case "scrum-uc":
            return SP_ITEMS;
         default:
            throw new IllegalArgumentException("Unknown case: " + caseId);
      }
   }

   private double uniform(final double[] range) {
      return range[0] + random.nextDouble() * (range[1] - range[0]);
   }

   private JSONObject sampleProperties(final int size) {
      final JSONObject properties = new JSONObject();
      switch(caseId) {
         case "cra":
            properties.put("attribute_share", uniform(CRA_ATTRIBUTE_SHARE));
            properties.put("dependency_density", uniform(CRA_DEPENDENCY_DENSITY));
            break;
         case "slb":
            properties.put("mean_load", uniform(SLB_MEAN_LOAD));
            break;
         case "scrum":
         case "scrum-uc":
            properties.put("stakeholders",
                  SP_STAKEHOLDERS[0] + random.nextInt(SP_STAKEHOLDERS[1] - SP_STAKEHOLDERS[0] + 1));
            properties.put("stakeholder_skew", uniform(SP_STAKEHOLDER_SKEW));
            final int minSprints = Math.max(1, (int) Math.round(uniform(SP_MIN_SPRINTS_PER_ITEM) * size));
            properties.put("min_sprints", minSprints);
            properties.put("max_sprints",
                  Math.max(minSprints + 1, (int) Math.round(uniform(SP_MAX_SPRINTS_PER_ITEM) * size)));
            break;
         default:
            throw new IllegalArgumentException("Unknown case: " + caseId);
      }
      return properties;
   }

   private EObject createModel(final int size, final JSONObject properties, final Random modelRandom) {
      switch(caseId) {
         case "cra":
            return createCra(size, properties, modelRandom);
         case "slb":
            return createSlb(size, properties, modelRandom);
         default:
            return createSp(size, properties, modelRandom);
      }
   }

   /** Lines per object, sorted: the hash does not depend on element or reference order. */
   private static String canonicalHash(final Resource resource) {
      final List<String> lines = new ArrayList<>();
      final TreeIterator<EObject> contents = resource.getAllContents();
      while(contents.hasNext()) {
         final EObject object = contents.next();
         final StringBuilder line = new StringBuilder(object.eClass().getName());
         for(final EAttribute attribute : object.eClass().getEAllAttributes()) {
            line.append('|').append(attribute.getName()).append('=').append(object.eGet(attribute));
         }
         for(final EReference reference : object.eClass().getEAllReferences()) {
            if(reference.isContainment() || reference.isContainer()) {
               continue;
            }
            final List<String> targets = new ArrayList<>();
            final Object value = object.eGet(reference);
            if(value instanceof List) {
               for(final Object target : (List<?>) value) {
                  targets.add(label((EObject) target));
               }
            } else if(value != null) {
               targets.add(label((EObject) value));
            }
            Collections.sort(targets);
            line.append('|').append(reference.getName()).append('=').append(targets);
         }
         lines.add(line.toString());
      }
      Collections.sort(lines);
      try {
         final MessageDigest digest = MessageDigest.getInstance("SHA-256");
         final byte[] hash = digest.digest(String.join("\n", lines).getBytes(StandardCharsets.UTF_8));
         final StringBuilder hex = new StringBuilder();
         for(final byte b : hash) {
            hex.append(String.format("%02x", b));
         }
         return hex.toString();
      } catch(final NoSuchAlgorithmException e) {
         throw new IllegalStateException(e);
      }
   }

   private static String label(final EObject object) {
      final StringBuilder label = new StringBuilder(object.eClass().getName());
      for(final EAttribute attribute : object.eClass().getEAllAttributes()) {
         label.append(':').append(object.eGet(attribute));
      }
      return label.toString();
   }

   private static ResourceSet newResourceSet() {
      ArchitecturePackage.eINSTANCE.eClass();
      StackPackage.eINSTANCE.eClass();
      SprintPlanningPackage.eINSTANCE.eClass();
      final ResourceSet resourceSet = new ResourceSetImpl();
      resourceSet.getResourceFactoryRegistry().getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());
      return resourceSet;
   }

   /** Each method uses each attribute, and calls each other method, with probability density. */
   private static EObject createCra(final int size, final JSONObject properties, final Random random) {
      final double attributeShare = properties.getDouble("attribute_share");
      final double density = properties.getDouble("dependency_density");
      final ArchitectureFactory factory = ArchitectureFactory.eINSTANCE;
      final ClassModel model = factory.createClassModel();

      // At least one of each, placed at random positions in the feature list.
      final int attributes = Math.max(1, Math.min(size - 1, (int) Math.round(size * attributeShare)));
      final List<Boolean> isAttribute = new ArrayList<>();
      for(int i = 0; i < size; i++) {
         isAttribute.add(i < attributes);
      }
      Collections.shuffle(isAttribute, random);

      final List<Method> methods = new ArrayList<>();
      final List<Attribute> attributeList = new ArrayList<>();
      for(int i = 0; i < size; i++) {
         final Feature feature;
         if(isAttribute.get(i)) {
            final Attribute attribute = factory.createAttribute();
            attributeList.add(attribute);
            feature = attribute;
            feature.setName("A_" + i);
         } else {
            final Method method = factory.createMethod();
            methods.add(method);
            feature = method;
            feature.setName("M_" + i);
         }
         model.getFeatures().add(feature);
      }

      for(final Method method : methods) {
         for(final Attribute attribute : attributeList) {
            if(random.nextDouble() < density) {
               method.getDataDependency().add(attribute);
            }
         }
         for(final Method other : methods) {
            if(other != method && random.nextDouble() < density) {
               method.getFunctionalDependency().add(other);
            }
         }
      }
      return model;
   }

   /** Stacks in a ring, with loads drawn uniformly around the requested mean load. */
   private static EObject createSlb(final int size, final JSONObject properties, final Random random) {
      final double meanLoad = properties.getDouble("mean_load");
      final int maxLoad = properties.optInt("max_load", 99);
      final double low = Math.max(0, 2 * meanLoad - maxLoad);
      final double high = Math.min(maxLoad, 2 * meanLoad);

      final StackFactory factory = StackFactory.eINSTANCE;
      final StackModel model = factory.createStackModel();
      final List<Stack> stacks = new ArrayList<>();
      for(int i = 0; i < size; i++) {
         final Stack stack = factory.createStack();
         stack.setId("Stack_" + i);
         stack.setLoad((int) Math.round(low + random.nextDouble() * (high - low)));
         stacks.add(stack);
         model.getStacks().add(stack);
      }
      for(int i = 0; i < size; i++) {
         stacks.get(i).setRight(stacks.get((i + 1) % size));
         stacks.get(i).setLeft(stacks.get(Math.floorMod(i - 1, size)));
      }
      return model;
   }

   /**
    * Work items with uniform effort and importance, assigned to stakeholders by weights.
    * Larger stakeholder_skew gives more uneven shares; every stakeholder gets at least one item.
    */
   private static EObject createSp(final int size, final JSONObject properties, final Random random) {
      final int numStakeholders = properties.getInt("stakeholders");
      final double skew = properties.getDouble("stakeholder_skew");
      final int maxValue = properties.optInt("max_value", 10);

      final SprintPlanningFactory factory = SprintPlanningFactory.eINSTANCE;
      final Plan plan = factory.createPlan();
      plan.setMinSprints(properties.getInt("min_sprints"));
      plan.setMaxSprints(properties.getInt("max_sprints"));

      final double[] weights = new double[numStakeholders];
      double totalWeight = 0;
      for(int i = 0; i < numStakeholders; i++) {
         plan.getStakeholders().add(factory.createStakeholder());
         weights[i] = Math.exp(skew * random.nextGaussian());
         totalWeight += weights[i];
      }

      final Backlog backlog = factory.createBacklog();
      plan.setBacklog(backlog);
      for(int i = 0; i < size; i++) {
         final WorkItem item = factory.createWorkItem();
         item.setName("W" + i);
         item.setEffort(1 + random.nextInt(maxValue));
         item.setImportance(1 + random.nextInt(maxValue));
         final Stakeholder stakeholder = plan.getStakeholders()
               .get(i < numStakeholders ? i : pickWeighted(weights, totalWeight, random));
         backlog.getWorkitems().add(item);
         item.setStakeholder(stakeholder);
      }
      return plan;
   }

   private static int pickWeighted(final double[] weights, final double total, final Random random) {
      double r = random.nextDouble() * total;
      for(int i = 0; i < weights.length; i++) {
         r -= weights[i];
         if(r <= 0) {
            return i;
         }
      }
      return weights.length - 1;
   }
}
