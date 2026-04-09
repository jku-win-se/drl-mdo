/*******************************************************************************
 * Copyright (c) 2015 Vienna University of Technology.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 * Martin Fleck (Vienna University of Technology) - initial API and implementation
 *
 * Initially developed in the context of ARTIST EU project www.artist-project.eu
 *******************************************************************************/
package at.ac.tuwien.big.momot.util;

import at.ac.tuwien.big.moea.search.algorithm.local.IFitnessComparator;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IEncodingStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.domain.IRewardStrategy;
import at.ac.tuwien.big.moea.search.algorithm.reinforcement.environment.ISolutionExtender;
import at.ac.tuwien.big.moea.search.fitness.MultiDimensionalFitnessFunction;
import at.ac.tuwien.big.moea.search.fitness.comparator.ConstraintFitnessComparator;
import at.ac.tuwien.big.moea.search.fitness.comparator.ObjectiveFitnessComparator;
import at.ac.tuwien.big.moea.util.CastUtil;
import at.ac.tuwien.big.momot.ModuleManager;
import at.ac.tuwien.big.momot.TransformationSearchOrchestration;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures.RewardFactory;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.domain.EncodingFactory;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.BaseEnvironment;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.SOEnvironment;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.environment.SolutionProvider;
import at.ac.tuwien.big.momot.search.engine.MomotEngine;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.search.solution.executor.SearchHelper;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.ToDoubleFunction;

import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.ecore.util.EcoreUtil.Copier;
import org.eclipse.emf.henshin.interpreter.Assignment;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.interpreter.Engine;
import org.eclipse.emf.henshin.interpreter.RuleApplication;
import org.eclipse.emf.henshin.interpreter.impl.EGraphImpl;
import org.eclipse.emf.henshin.model.Parameter;
import org.eclipse.emf.henshin.model.ParameterKind;
import org.eclipse.emf.henshin.model.resource.HenshinResourceSet;
import org.moeaframework.core.Population;
import org.moeaframework.core.Solution;

public final class MomotUtil {

   private static List<ParameterKind> notExtractParameterKinds = List.of(ParameterKind.VAR);

   public static <T extends Solution> Iterable<T> asIterables(final Population population, final Class<T> clazz) {
      final List<T> solutions = new ArrayList<>();
      for(final Solution solution : population) {
         solutions.add(CastUtil.asClass(solution, clazz));
      }
      return solutions;
   }

   public static TransformationSolution assertTransformationSolution(final Solution solution) {
      return CastUtil.assertClass(solution, TransformationSolution.class);
   }

   public static TransformationSolution asTransformationSolution(final Solution solution) {
      return CastUtil.asClass(solution, TransformationSolution.class);
   }

   public static EGraph copy(final EGraph original) {
      if(original == null) {
         return null;
      }

      try {
         // Copy the resource: (necessary?)
         // List<EObject> roots = original.getRoots();
         // EObject root = roots.get(0);
         // Resource resource = root.eResource();
         // Copier copier = null;
         // if(resource != null) {
         // Resource copiedResource = new ResourceImpl();
         // copier = new Copier();
         // copiedResource.getContents().addAll(copier.copyAll(resource.getContents()));
         // copier.copyReferences();
         // }

         // Copy the graph:
         return original.copy(null);
      } catch(final Exception e) {
         // System.out.println("MomotUtil.copy(): " + e.getClass().getSimpleName() + ": " + e.getMessage());
         // System.out.flush();
         try {
            return original.copy(null);
         } catch(final Exception ex) {

            // System.out.println("MomotUtil.copy(): " + e.getClass().getSimpleName() + ": " + e.getMessage());
            // System.out.flush();
            final Copier copier = new Copier();
            copier.copyAll(original.getRoots());
            copier.copyReferences();

            Boolean nullObjects = false;
            final EGraph copy = new EGraphImpl(original.size());
            for(final EObject object : original) {
               final EObject objectCopy = copier.get(object);
               if(objectCopy == null) {
                  nullObjects = true;
                  break;
               }
               copy.add(objectCopy);
            }

            if(!nullObjects) {
               return copy;
            } else {
               return new EGraphImpl(EcoreUtil.copyAll(original.getRoots()));
            }
         }
      }
   }

   public static EGraph createEGraph(final EObject root) {
      return new EGraphImpl(root);
   }

   public static EGraph createEGraph(final Resource resource) {
      return new EGraphImpl(resource);
   }

   public static HenshinResourceSet createResourceSet() {
      return new HenshinResourceSet();
   }

   public static HenshinResourceSet createResourceSet(final String baseDir) {
      return new HenshinResourceSet(baseDir);
   }

   public static EGraph eGraphOf(final Resource resource, final boolean decoupleFromResource) {
      final EGraph g = MomotUtil.createEGraph(resource);
      if(decoupleFromResource) {
         return MomotUtil.copy(g);
      }
      return g;
   }

   public static Map<String, Object> extractParameters(final Assignment assignment) {
      final Map<String, Object> paramValues = new HashMap<>();

      for(final Parameter p : assignment.getUnit().getParameters()) {
         if(assignment.getParameterValue(p) != null) {
            if(!notExtractParameterKinds.contains(p.getKind())) {
               paramValues.put(p.getName(), assignment.getParameterValue(p));
            }
         }
      }

      return paramValues;

   }

   public static Map<String, Object> extractParameters(final List<String> paramList, final Assignment assignment) {
      final Map<String, Object> paramValues = new HashMap<>();

      for(final Parameter p : assignment.getUnit().getParameters()) {
         if(assignment.getParameterValue(p) != null) {
            if(!notExtractParameterKinds.contains(p.getKind())
                  && (paramList == null || paramList.contains(p.getName()))) {
               paramValues.put(p.getName(), assignment.getParameterValue(p));

            }
         }
      }

      return paramValues;
   }

   public static Map<String, Object> extractParameters(final List<String> paramList, final RuleApplication... raps) {
      final Map<String, Object> paramValues = new HashMap<>();

      for(final RuleApplication ra : raps) {
         for(final Parameter p : ra.getUnit().getParameters()) {
            if(ra.getPartialMatch().getParameterValue(p) != null) {
               if(!notExtractParameterKinds.contains(p.getKind())
                     && (paramList == null || paramList.contains(p.getName()))) {
                  paramValues.put(p.getName(), ra.getPartialMatch().getParameterValue(p));
               }
            }
         }

      }
      return paramValues;
   }

   public static Map<String, Object> extractParameters(final RuleApplication... raps) {
      final Map<String, Object> paramValues = new HashMap<>();

      for(final RuleApplication ra : raps) {
         for(final Parameter p : ra.getUnit().getParameters()) {
            if(ra.getPartialMatch().getParameterValue(p) != null) {
               if(!notExtractParameterKinds.contains(p.getKind())) {
                  paramValues.put(p.getName(), ra.getPartialMatch().getParameterValue(p));
               }
            }
         }

      }
      return paramValues;
   }

   public static EObject getRoot(final EGraph graph) {
      return graph.getRoots().get(0);
   }

   public static <T extends EObject> T getRoot(final EGraph graph, final Class<T> rootClass) {
      return CastUtil.asClass(getRoot(graph), rootClass);
   }

   public static <E extends IEncodingStrategy, R extends IRewardStrategy> BaseEnvironment<TransformationSolution> initBaseLearnEnv(
         final ModuleManager moduleManager, final IEGraphMultiDimensionalFitnessFunction fitnessFunction,
         final String objectiveName, final String constraintName, final Class<R> rewardStrategy,
         final Class<E> encodingStrategy) {

      final IEncodingStrategy<TransformationSolution> enc = encodingStrategy != null
            ? EncodingFactory.createEncoder(encodingStrategy)
            : null;
      final IRewardStrategy<TransformationSolution> rewStrat = rewardStrategy != null
            ? RewardFactory.createRewardStrategy(rewardStrategy)
            : null;

      final IFitnessComparator<?, Solution> constrFitComparator = fitnessFunction
            .getConstraintIndex(constraintName) >= 0
                  ? new ConstraintFitnessComparator<>(fitnessFunction.getConstraintIndex(constraintName))
                  : null;
      final BaseEnvironment<TransformationSolution> env = new BaseEnvironment<>(fitnessFunction,
            new ObjectiveFitnessComparator<>(fitnessFunction.getObjectiveIndex(objectiveName)), constrFitComparator,
            objectiveName, constraintName, rewStrat, enc);
      try {
         env.setEvaluationMethod(fitnessFunction, MultiDimensionalFitnessFunction.class.getMethod("evaluate"));
      } catch(NoSuchMethodException | SecurityException e) {
         e.printStackTrace();
      }

      // final HenshinResourceSet hrs = new HenshinResourceSet();

      // final EGraph initialGraph = MomotUtil.loadGraph(hrs, INPUT_MODEL);

      final Engine engine = new MomotEngine(true, new String());

      final SearchHelper searchHelper = new SearchHelper(engine, moduleManager);
      // final TransformationSolution ts = searchHelper.createEmptyTransformationSolution(initialGraph, 3,
      // fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size());

      final ISolutionExtender<TransformationSolution> solutionExtender = new SolutionProvider<>(searchHelper);

      env.setSolutionProvider(solutionExtender);

      return env;
   }

   public static <E extends IEncodingStrategy, R extends IRewardStrategy> BaseEnvironment<TransformationSolution> initBaseLearnEnv(
         final TransformationSearchOrchestration orchestration, final String objectiveName,
         final Class<R> rewardStrategy, final Class<E> encodingStrategy) {
      final ModuleManager moduleManager = orchestration.getModuleManager();
      final IEGraphMultiDimensionalFitnessFunction fitnessFunction = orchestration.getFitnessFunction();

      final IEncodingStrategy<TransformationSolution> enc = EncodingFactory.createEncoder(encodingStrategy);
      final IRewardStrategy<TransformationSolution> rewStrat = rewardStrategy != null
            ? RewardFactory.createRewardStrategy(rewardStrategy)
            : null;
      final BaseEnvironment<TransformationSolution> env = new BaseEnvironment<>(fitnessFunction,
            new ObjectiveFitnessComparator<>(fitnessFunction.getObjectiveIndex(objectiveName)), null, objectiveName,
            null, rewStrat, enc);

      // final HenshinResourceSet hrs = new HenshinResourceSet();

      // final EGraph initialGraph = MomotUtil.loadGraph(hrs, INPUT_MODEL);

      final Engine engine = new MomotEngine(true, new String());

      final SearchHelper searchHelper = new SearchHelper(engine, moduleManager);
      // final TransformationSolution ts = searchHelper.createEmptyTransformationSolution(initialGraph, 3,
      // fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size());

      final ISolutionExtender<TransformationSolution> solutionExtender = new SolutionProvider<>(searchHelper);

      env.setSolutionProvider(solutionExtender);

      return env;
   }

   public static <E extends IEncodingStrategy, R extends IRewardStrategy> SOEnvironment<TransformationSolution> initSOEnv(
         final ModuleManager moduleManager, final IEGraphMultiDimensionalFitnessFunction fitnessFunction,
         final ToDoubleFunction<double[]> fitnessAggregator, final ToDoubleFunction<double[]> constraintAggregator,
         final ToDoubleFunction<EObject> fitnessScaler, final ToDoubleFunction<EObject> constraintScaler,
         final Class<R> rewardStrategy, final Class<E> encodingStrategy) {

      final IEncodingStrategy<TransformationSolution> enc = encodingStrategy != null
            ? EncodingFactory.createEncoder(encodingStrategy)
            : null;
      final IRewardStrategy<TransformationSolution> rewStrat = rewardStrategy != null
            ? RewardFactory.createRewardStrategy(rewardStrategy)
            : null;

      final SOEnvironment<TransformationSolution> env = new SOEnvironment<>(fitnessAggregator, constraintAggregator,
            fitnessScaler, constraintScaler, rewStrat, enc);
      try {
         env.setEvaluationMethod(fitnessFunction,
               MultiDimensionalFitnessFunction.class.getMethod("evaluate", Solution.class));
      } catch(NoSuchMethodException | SecurityException e) {
         e.printStackTrace();
      }

      // final HenshinResourceSet hrs = new HenshinResourceSet();

      // final EGraph initialGraph = MomotUtil.loadGraph(hrs, INPUT_MODEL);

      final Engine engine = new MomotEngine(true, new String());

      final SearchHelper searchHelper = new SearchHelper(engine, moduleManager);
      // final TransformationSolution ts = searchHelper.createEmptyTransformationSolution(initialGraph, 3,
      // fitnessFunction.getObjectiveNames().size(), fitnessFunction.getConstraintNames().size());

      final ISolutionExtender<TransformationSolution> solutionExtender = new SolutionProvider<>(searchHelper);

      env.setSolutionProvider(solutionExtender);

      return env;
   }

   public static EGraph loadGraph(final HenshinResourceSet set, final String resourceUri) {
      final Resource model = set.getResource(resourceUri);
      return new EGraphImpl(model);
   }

   public static EGraph loadGraph(final String resourceUri) {
      return loadGraph(createResourceSet(), resourceUri);
   }

   public static TransformationSolution readInitialSolution(final EGraph initialGraph, final int nrOfObjectives,
         final int nrOfConstraints) {

      final TransformationSolution ts = new TransformationSolution(MomotUtil.copy(initialGraph), 0, nrOfObjectives,
            nrOfConstraints);
      return ts;
   }

   public static List<TransformationSolution> readInitialSolutions(final List<File> fileList, final int nrOfObjectives,
         final int nrOfConstraints) {
      throw new RuntimeException("TODO adapt to new graph parameter isntead of filelist");
      // final List<TransformationSolution> initialSolutions = new ArrayList<>();
      // for(final File f : fileList) {
      // initialSolutions.add(readInitialSolution(f, nrOfObjectives, nrOfConstraints));
      // }
      // return initialSolutions;
   }

   public static void saveGraph(final EGraph graph, final String targetResourceUri) {
      saveGraph(createResourceSet(), graph, targetResourceUri);
   }

   public static void saveGraph(final HenshinResourceSet set, final EGraph graph, final String targetResource) {
      if(targetResource != null) {
         set.saveEObject(getRoot(graph), targetResource);
      }
   }

   private MomotUtil() {
   }
}
