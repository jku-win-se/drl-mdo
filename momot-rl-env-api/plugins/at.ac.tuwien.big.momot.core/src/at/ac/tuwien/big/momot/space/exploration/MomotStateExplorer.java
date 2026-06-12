package at.ac.tuwien.big.momot.space.exploration;

import at.ac.tuwien.big.momot.ds.ApplicationState;
import at.ac.tuwien.big.momot.problem.solution.variable.ITransformationVariable;
import at.ac.tuwien.big.momot.problem.solution.variable.RuleApplicationVariable;
import at.ac.tuwien.big.momot.search.engine.MomotEngine;
import at.ac.tuwien.big.momot.util.MomotUtil;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptException;

import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.interpreter.Engine;
import org.eclipse.emf.henshin.interpreter.InterpreterFactory;
import org.eclipse.emf.henshin.interpreter.Match;
import org.eclipse.emf.henshin.interpreter.RuleApplication;
import org.eclipse.emf.henshin.model.Node;
import org.eclipse.emf.henshin.model.Rule;
import org.eclipse.emf.henshin.statespace.EqualityHelper;
import org.eclipse.emf.henshin.statespace.Model;
import org.eclipse.emf.henshin.statespace.Path;
import org.eclipse.emf.henshin.statespace.State;
import org.eclipse.emf.henshin.statespace.StateSpace;
import org.eclipse.emf.henshin.statespace.StateSpaceException;
import org.eclipse.emf.henshin.statespace.StateSpaceFactory;
import org.eclipse.emf.henshin.statespace.StateSpaceIndex;
import org.eclipse.emf.henshin.statespace.StateSpacePlugin;
import org.eclipse.emf.henshin.statespace.StateValidator;
import org.eclipse.emf.henshin.statespace.Transition;
import org.eclipse.emf.henshin.statespace.Validator;
import org.eclipse.emf.henshin.statespace.hashcodes.StateSpaceHashCodeUtil;
import org.eclipse.emf.henshin.statespace.util.ObjectKeyHelper;

public class MomotStateExplorer {

   public class ExplorationStepResponse {
      public List<Transition> transitions;
      public List<List<List<ITransformationVariable>>> transitionsTransformationStates;

      public ExplorationStepResponse(final List<Transition> transitions,
            final List<List<List<ITransformationVariable>>> transitionsTrafoStates) {
         this.transitions = transitions;
         this.transitionsTransformationStates = transitionsTrafoStates;

      }
   }

   /**
    * Model post processor implementation.
    *
    * @author Chrstian Krause
    */
   class ModelPostProcessor {

      // Script engine:
      private final ScriptEngine engine;

      // Cached post processing script:
      private String script;

      /**
       * Constructor.
       *
       * @param statepace
       *           State space.
       */
      public ModelPostProcessor(final StateSpace stateSpace) {
         final ScriptEngineManager manager = new ScriptEngineManager();
         engine = manager.getEngineByName("JavaScript");
         script = stateSpace.getProperties().get("postProcessor");
         if(script != null && script.trim().length() == 0) {
            script = null;
         }
         if(script != null) {
            final String imports = "importPackage(java.lang);\n" + "importPackage(java.util);\n"
                  + "importPackage(org.eclipse.emf.ecore);\n";
            script = imports + script;
         }
      }

      /**
       * Do the post processing for a model.
       *
       * @param model
       *           Model.
       * @throws StateSpaceException
       *            On execution errors.
       */
      public void process(final Model model) throws StateSpaceException {
         if(script != null) {
            final EObject root = model.getResource().getContents().get(0);
            synchronized(engine) {
               engine.put("model", root);
               try {
                  engine.eval(script);
               } catch(final ScriptException e) {
                  throw new StateSpaceException(e);
               }
            }
         }
      }

   }

   /**
    * Dummy progress monitor.
    */
   private static final NullProgressMonitor NULL_PROGRESS_MONITOR = new NullProgressMonitor();

   /*
    * Get the priority property key for a rule.
    */
   private static String getRulePriorityKey(final Rule rule) {
      String key = rule.getName();
      if(key == null) {
         key = "";
      }
      key = key.replaceAll(" ", "_");
      key = key.replaceAll("\t", "_");
      key = key.replaceAll("\n", "_");
      final String first = key.substring(0, 1).toUpperCase();
      if(key.length() == 1) {
         key = first;
      } else {
         key = first + key.substring(1);
      }
      return "priority" + key;
   }

   /**
    * State space index.
    */
   private final StateSpaceIndex index;

   /**
    * Cached state space.
    */
   private final StateSpace stateSpace;

   /**
    * Cached equality helper.
    */
   private final EqualityHelper equalityHelper;

   /**
    * Cached engine.
    */
   private final Engine engine;

   /**
    * Cached result.
    */
   private final List<Transition> result;

   private final List<List<List<ITransformationVariable>>> transitionTrafos;

   /**
    * Whether to use object keys.
    */
   private final boolean useObjectKeys;

   /**
    * Cached rules sorted by priority.
    */
   private final Rule[][] rules;

   /**
    * Cached rule parameters.
    */
   private final Map<Rule, List<Node>> ruleParameters;

   /**
    * Cached rule application.
    */
   private final RuleApplication application;

   /**
    * Cached identity types.
    */
   private final EList<EClass> identityTypes;

   /**
    * Cached post-processor.
    */
   private final ModelPostProcessor postProcessor;

   /**
    * Cached state validator for deciding whether a state is a goal state.
    */
   private StateValidator goalStateValidator;

   /**
    * Cached flag determining whether to collect missing roots.
    */
   private final boolean collectMissingRoots;

   Map<Integer, List<ApplicationState>> stateToRuleApps;
   Map<Integer, List<List<ITransformationVariable>>> stateToTrafoVars;

   /**
    * Default constructor.
    */
   public MomotStateExplorer(final StateSpaceIndex index, final Map<Integer, List<ApplicationState>> stateToRuleApps,
         final Map<Integer, List<List<ITransformationVariable>>> stateToTrafoVars) {

      // Set the index:
      this.index = index;

      this.stateToRuleApps = stateToRuleApps;
      this.stateToTrafoVars = stateToTrafoVars;

      // Cache basic data:
      stateSpace = index.getStateSpace();
      equalityHelper = stateSpace.getEqualityHelper();
      result = new ArrayList<>();
      transitionTrafos = new ArrayList<>();
      identityTypes = equalityHelper.getIdentityTypes();
      useObjectKeys = true;

      // Sort the rules by their priorities:
      final Map<Integer, List<Rule>> prioritizedRules = new LinkedHashMap<>();
      for(final Rule rule : stateSpace.getRules()) {
         int priority = 0;
         final String key = getRulePriorityKey(rule);
         String val = stateSpace.getProperties().get(key);
         if(val == null) {
            val = stateSpace.getProperties().get(key.toLowerCase());
         }
         if(val != null && val.trim().length() != 0) {
            try {
               priority = Integer.parseInt(val.trim());
            } catch(final Throwable t) {
               StateSpacePlugin.INSTANCE.logError("Error parsing priority of rule " + rule.getName(), t);
            }
         }
         if(!prioritizedRules.containsKey(priority)) {
            prioritizedRules.put(priority, new ArrayList<Rule>());
         }
         prioritizedRules.get(priority).add(rule);
      }
      final List<Integer> priorities = new ArrayList<>(prioritizedRules.keySet());
      Collections.sort(priorities);
      Collections.reverse(priorities);

      // Cache rule data:
      rules = new Rule[priorities.size()][];
      for(int i = 0; i < rules.length; i++) {
         rules[i] = prioritizedRules.get(priorities.get(i)).toArray(new Rule[0]);
      }
      ruleParameters = new HashMap<>();
      if(useObjectKeys) {
         for(final Rule rule : stateSpace.getRules()) {
            ruleParameters.put(rule, rule.getParameterNodes());
         }
      }

      // Collect missing roots flag:
      final String collect = stateSpace.getProperties().get(StateSpace.PROPERTY_COLLECT_MISSING_ROOTS);
      collectMissingRoots = collect != null && ("true".equalsIgnoreCase(collect) || "yes".equalsIgnoreCase(collect));

      // Set-up the engine:
      engine = new MomotEngine(true);
      engine.getOptions().put(Engine.OPTION_DETERMINISTIC, Boolean.TRUE); // really make sure it is deterministic

      // Create a rule application:
      application = InterpreterFactory.INSTANCE.createRuleApplication(engine);

      // Set-up the goal state validator:
      goalStateValidator = null;
      String goalProperty = stateSpace.getProperties().get(StateSpace.PROPERTY_GOAL_PROPERTY);
      if(goalProperty != null && goalProperty.trim().length() > 0) {
         String type = null;
         if(goalProperty.indexOf(':') > 0) {
            type = goalProperty.substring(0, goalProperty.indexOf(':')).trim();
            goalProperty = goalProperty.substring(goalProperty.indexOf(':') + 1).trim();
         }
         for(final Validator validator : StateSpacePlugin.INSTANCE.getValidators().values()) {
            if(validator instanceof StateValidator && (type == null || type.equalsIgnoreCase(validator.getName()))) {
               try {
                  goalStateValidator = (StateValidator) validator.getClass().newInstance();
                  goalStateValidator.setStateSpaceIndex(index);
                  goalStateValidator.setProperty(goalProperty);
                  break;
               } catch(final Exception e) {
               }
            }
         }
         if(goalStateValidator == null) {
            StateSpacePlugin.INSTANCE.logError("Error loading goal state validator: " + type, null);
         }
      }

      // Post-processor:
      postProcessor = new ModelPostProcessor(stateSpace);

   }

   /**
    * Derive a model using a path and a given start model.
    *
    * @param path
    *           Path.
    * @param sourceModel
    *           Source model.
    * @return The derived model.
    * @throws StateSpaceException
    *            On errors.
    */
   public Model deriveModel(final Path path, Model sourceModel) throws StateSpaceException {

      // We need to copy the start model!!!
      sourceModel = sourceModel.getCopy(null);
      application.setEGraph(sourceModel.getEGraph());
      Match match;

      // Derive the model for the current state:
      for(final Transition transition : path) {

         // Find the match:
         match = getMatch(transition, sourceModel);

         // Apply the rule with the found match:
         application.setCompleteMatch(match);
         if(!application.execute(null)) {
            throw new StateSpaceException("Error deriving model");
         }
         postProcessor.process(sourceModel);
         if(collectMissingRoots) {
            sourceModel.collectMissingRootObjects();
         }

         // Debug: Validate model if necessary:
         // int hashCode = getStateSpace().getEqualityHelper().hashCode(model);
         // if (transition.getTarget().getHashCode()!=hashCode) {
         // throw new StateSpaceException("Error constructing model for state " +
         // transition.getTarget().getIndex() + " in path " + trace);
         // }

         // Update object keys if necessary:
         if(useObjectKeys) {
            sourceModel.updateObjectKeys(identityTypes);
         }

      }

      // Set the object keys if necessary:
      // if (useObjectKeys) {
      // model.setObjectKeys(path.getTarget().getObjectKeys());
      // }

      // Update the object hash codes:
      StateSpaceHashCodeUtil.computeHashCode(sourceModel, equalityHelper);

      // Done.
      return sourceModel;

   }

   /**
    * Derive a model.
    *
    * @param State
    *           state.
    * @param fromInitial
    *           Whether to derive it from an initial state.
    * @return The derived model.
    * @throws StateSpaceException
    *            On errors.
    */
   public Model deriveModel(final State state, final boolean fromInitial) throws StateSpaceException {

      // Find a path from one of the states predecessors:
      final Path trace = new Path();
      State source = state;
      State target;
      Model start = null;
      final List<State> states = index.getStateSpace().getStates();
      try {
         while(start == null) {
            target = source;
            source = states.get(target.getDerivedFrom());
            trace.addFirst(source.getOutgoing(target, null, -1, null));
            start = index.getCachedModel(source);
            if(fromInitial && !source.isInitial()) {
               start = null;
            }
         }
      } catch(final Throwable t) {
         throw new StateSpaceException("Error deriving model for " + state, t);
      }

      // Now derive the model:
      return deriveModel(trace, start);

   }

   /**
    * Explore a state without actually changing the state space.
    * This method does not check if the state is explored already
    * or whether any of the transitions or states exists already.
    *
    * @param state
    *           State to be explored.
    * @throws StateSpaceException
    *            On explore errors.
    */
   public ExplorationStepResponse doExplore(final State state) throws StateSpaceException {

      // System.out.println("\nExplore from state " + state.getIndex() + "\n");
      // Clear the result:
      result.clear();
      transitionTrafos.clear();

      // Get the state model and create an engine for it:
      final Model model = index.getModel(state);
      final EGraph graph = model.getEGraph();

      // Find the first layer with an applicable rule and apply all its rules:
      for(final Rule[] rule : rules) {

         // Remember whether we applied at least one rule in the layer:
         boolean applicable = false;

         // Try all rules in the current layer:
         for(final Rule element : rule) {

            // Get the parameters of the rule:
            final List<Node> parameters = ruleParameters.get(element);

            // Iterate over all matches:
            int matchIndex = 0;
            for(final Match match : engine.findMatches(element, graph, null)) {

               // We know it is applicable:
               applicable = true;

               // Transform the model:
               final Model transformed = model.getCopy(match);
               application.setRule(element);
               application.setEGraph(transformed.getEGraph());
               application.setCompleteMatch(match);
               if(!application.execute(null)) {
                  throw new StateSpaceException("Error applying rule \"" + element.getName()
                        + "\" to found match in state " + state.getIndex());
               }

               final List<ApplicationState> curStateApplications = stateToRuleApps.get(state.getHashCode());
               final ApplicationState curAppState = ExploreUtils.unitApplicationToApplicationState(application);
               final List<List<ITransformationVariable>> emergingStateTrafoPaths = new ArrayList<>();

               if(!curStateApplications.contains(curAppState)) {
                  curStateApplications.add(curAppState);

                  final ITransformationVariable newVar = new RuleApplicationVariable(engine, MomotUtil.copy(graph),
                        element, match);
                  for(final List<ITransformationVariable> vars : stateToTrafoVars.get(state.getHashCode())) {
                     final List<ITransformationVariable> newTrafoForState = new ArrayList<>(vars);
                     newTrafoForState.add(newVar);

                     emergingStateTrafoPaths.add(newTrafoForState);
                  }
                  // System.out.println(emergingStateTrafoPaths);

               }

               postProcessor.process(transformed);
               if(collectMissingRoots) {
                  transformed.collectMissingRootObjects();
               }

               // Create a new state:
               final State newState = StateSpaceFactory.eINSTANCE.createState();
               newState.setModel(transformed);

               // Update object keys if necessary:
               if(useObjectKeys) {
                  transformed.updateObjectKeys(identityTypes);
                  final int[] objectKeys = transformed.getObjectKeys();
                  newState.setObjectKeys(objectKeys);
                  newState.setObjectCount(objectKeys.length);
               }

               // Now compute and set the hash code (after the node IDs have been updated!):
               final int newHashCode = equalityHelper.hashCode(transformed);
               newState.setHashCode(newHashCode);
               newState.setDerivedFrom(state.getIndex());

               // Create a new transition:
               final Transition newTransition = StateSpaceFactory.eINSTANCE.createTransition();
               newTransition.setRule(element);
               newTransition.setMatch(matchIndex);
               newTransition.setTarget(newState);

               // Set the parameters if necessary:
               if(useObjectKeys) {
                  final int[] params = new int[parameters.size()];
                  for(int p = 0; p < params.length; p++) {
                     final Node node = parameters.get(p);
                     EObject matched = match.getNodeTarget(node);
                     if(matched == null) {
                        matched = application.getResultMatch().getNodeTarget(node);
                     }
                     final int objectKey = transformed.getObjectKeysMap().get(matched);
                     params[p] = ObjectKeyHelper.createObjectKey(matched.eClass(), objectKey, identityTypes);
                  }
                  newTransition.setParameterKeys(params);
                  newTransition.setParameterCount(params.length);
               }

               // Remember the transition:
               result.add(newTransition);
               transitionTrafos.add(emergingStateTrafoPaths);
               // Increase the match index:
               matchIndex++;

            }
         }

         // If at least one rule in the layer was applicable, we stop:
         if(applicable) {
            break;
         }

      }

      // Done.
      return new ExplorationStepResponse(result, transitionTrafos);

   }

   /**
    * Get the last result match used during the derivation of a model.
    *
    * @return The last used result match.
    */
   public Match getLastResultMatch() {
      return application.getResultMatch();
   }

   /**
    * Get the match for a given transition.
    *
    * @param transition
    *           The transition.
    * @param sourceModel
    *           The model of the source state.
    * @return The match.
    * @throws StateSpaceException
    *            On state space errors.
    */
   public Match getMatch(final Transition transition, final Model sourceModel) throws StateSpaceException {
      final EGraph graph = sourceModel.getEGraph();
      application.setEGraph(graph);
      application.setRule(transition.getRule());
      final int targetMatch = transition.getMatch();
      int currentMatch = 0;
      for(final Match foundMatch : engine.findMatches(transition.getRule(), graph, null)) {
         if(currentMatch++ == targetMatch) {
            return foundMatch;
         }
      }
      throw new StateSpaceException("Illegal transition in state " + transition.getSource());
   }

   /*
    * Perform a sanity check for the exploration. For testing only.
    * This check if doExplore() really yields equal results when invoked
    * more than once on the same state.
    */
   /*
    * protected void checkEngineDeterminism(State state) throws StateSpaceException {
    * // Explore the state without changing the state space:
    * List<Transition> transitions = acquireTransitionList();
    * doExplore(state, transitions);
    * // Do it again and compare the results.
    * for (int i=0; i<25; i++) {
    * List<Transition> transitions2 = acquireTransitionList();
    * doExplore(state, transitions2);
    * if (transitions.size()!=transitions2.size()) {
    * throw new StateSpaceException("Sanity check 1 failed!");
    * }
    * for (int j=0; j<transitions.size(); j++) {
    * Transition t1 = transitions.get(j);
    * Transition t2 = transitions2.get(j);
    * if (t1.getRule()!=t2.getRule() || t1.getMatch()!=t2.getMatch()) {
    * throw new StateSpaceException("Sanity check 2 failed!");
    * }
    * State s1 = t1.getTarget();
    * State s2 = t2.getTarget();
    * if (s1.getHashCode()!=s2.getHashCode()) {
    * throw new StateSpaceException("Sanity check 3 failed!");
    * }
    * if (!getStateSpace().getEqualityHelper().equals(s1.getModel(),s2.getModel())) {
    * throw new StateSpaceException("Sanity check 4 failed!");
    * }
    * }
    * }
    * }
    */

   /**
    * Check whether a state is a goal state.
    *
    * @param state
    *           State to be checked.
    * @result <code>true</code> if it is a goal state.
    */
   public boolean isGoalState(final State state) throws StateSpaceException {
      if(goalStateValidator == null) {
         return false;
      }
      try {
         return goalStateValidator.validate(state, NULL_PROGRESS_MONITOR).isValid();
      } catch(final Throwable e) {
         throw new StateSpaceException(e);
      }
   }

}
