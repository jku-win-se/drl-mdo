package at.ac.tuwien.big.momot.space.exploration;

import at.ac.tuwien.big.moea.search.fitness.IFitnessEvaluation;
import at.ac.tuwien.big.momot.TransformationStateExplorer.MyInterface;
import at.ac.tuwien.big.momot.ds.ApplicationState;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.problem.solution.variable.ITransformationVariable;
import at.ac.tuwien.big.momot.search.fitness.IEGraphMultiDimensionalFitnessFunction;
import at.ac.tuwien.big.momot.space.exploration.MomotStateExplorer.ExplorationStepResponse;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.ConcurrentHashMap;

import org.eclipse.emf.henshin.interpreter.EGraph;
import org.eclipse.emf.henshin.model.Rule;
import org.eclipse.emf.henshin.statespace.Model;
import org.eclipse.emf.henshin.statespace.State;
import org.eclipse.emf.henshin.statespace.StateSpace;
import org.eclipse.emf.henshin.statespace.StateSpaceException;
import org.eclipse.emf.henshin.statespace.Transition;
import org.eclipse.emf.henshin.statespace.impl.BasicStateSpaceManager;
import org.moeaframework.core.Population;

public class MomotStateSpaceManager extends BasicStateSpaceManager {
   public class CustomMap<K, V> implements Map<K, V> {
      Map<K, V> wMap;

      public CustomMap() {
         wMap = new ConcurrentHashMap<>();
      }

      @Override
      public void clear() {
         wMap.clear();
      }

      @Override
      public boolean containsKey(final Object key) {
         return wMap.containsKey(key);
      }

      @Override
      public boolean containsValue(final Object value) {
         return wMap.containsValue(value);
      }

      @Override
      public Set<Entry<K, V>> entrySet() {
         return wMap.entrySet();
      }

      @Override
      public V get(final Object key) {
         return wMap.get(key);
      }

      @Override
      public boolean isEmpty() {
         return wMap.isEmpty();
      }

      @Override
      public Set<K> keySet() {
         return wMap.keySet();
      }

      @Override
      public V put(final K key, final V value) {
         for(final List<ITransformationVariable> chain : (List<List<ITransformationVariable>>) value) {
            final TransformationSolution ts = new TransformationSolution(initialGraph, chain,
                  function.getObjectiveNames().size(), function.getConstraintNames().size());
            ts.execute();
            function.evaluate(ts);
            if(ts.getObjectives()[0] != IFitnessEvaluation.WORST_FITNESS) {
               // monitor.info("New solution found",
               // solutionReprFunction != null ? String.format(solutionReprFunction.solutionRepresentation(ts))
               // : "");
               population.add(ts);
            }

         }
         return wMap.put(key, value);
      }

      @Override
      public void putAll(final Map<? extends K, ? extends V> m) {
         wMap.putAll(m);
      }

      @Override
      public V remove(final Object key) {
         return wMap.remove(key);
      }

      @Override
      public int size() {
         return wMap.size();
      }

      @Override
      public Collection<V> values() {
         return wMap.values();
      }

   }

   private final Population population;
   private final ExplorationMonitor monitor;
   /**
    * State exploration helpers.
    */
   protected final Stack<MomotStateExplorer> explorers = new Stack<>();

   protected final List<Integer> indices;
   protected Map<Integer, List<ApplicationState>> stateToRuleApps;

   protected Map<Integer, List<List<ITransformationVariable>>> stateToTrafoVars;
   protected IEGraphMultiDimensionalFitnessFunction function;
   protected EGraph initialGraph;
   protected MyInterface solutionReprFunction;

   public MomotStateSpaceManager(final StateSpace stateSpace, final Model modelInstance, final EGraph initialGraph,
         final IEGraphMultiDimensionalFitnessFunction function, final ExplorationMonitor monitor) {
      super(stateSpace);
      this.initialGraph = initialGraph;
      this.monitor = monitor;
      this.function = function;
      this.population = new Population();
      this.stateToRuleApps = new CustomMap<>();
      this.stateToTrafoVars = new CustomMap<>();
      try {
         this.createInitialState(modelInstance);
      } catch(final StateSpaceException e) {
         throw new RuntimeException("Cannot create initial state!");
      }

      final int initialStateIdx = stateSpace.getInitialStates().get(0).getHashCode();
      final List<ITransformationVariable> initStateSolutionVars = new ArrayList<>();

      this.stateToRuleApps.put(initialStateIdx, new ArrayList<>());
      this.stateToTrafoVars.put(initialStateIdx, new ArrayList<>());
      this.stateToTrafoVars.get(initialStateIdx).add(initStateSolutionVars);

      this.indices = new ArrayList<>();
   }

   protected MomotStateExplorer acquireStateExplorer() {
      synchronized(explorers) {
         try {
            return explorers.pop();
         } catch(final Throwable t) {
            return new MomotStateExplorer(this, stateToRuleApps, stateToTrafoVars);
         }
      }
   }

   /*
    * (non-Javadoc)
    * @see
    * org.eclipse.emf.henshin.statespace.impl.StateSpaceIndexImpl#deriveModel(org.eclipse.emf.henshin.statespace.State,
    * boolean)
    */
   @Override
   protected Model deriveModel(final State state, final boolean fromInitial) throws StateSpaceException {
      final MomotStateExplorer explorer = acquireStateExplorer();
      final Model model = explorer.deriveModel(state, fromInitial);
      releaseExplorer(explorer);
      return model;
   }

   /*
    * Acquire a state explorer.
    */

   /**
    * Explore a given state.
    *
    * @param state
    *           State to be explored.
    * @param generateLocation
    *           Whether to generate locations for the new state.
    * @return List of newly created successor states.
    * @throws StateSpaceException
    *            On errors.
    */
   @Override
   protected List<State> exploreState(final State state, final boolean generateLocation) throws StateSpaceException {
      // System.out.println(state.getIndex());
      // if(indices.contains(state.getIndex())) {
      // System.out.println(indices.indexOf(state.getIndex()));
      // }
      /*
       * if(state.getModel() != null) {
       * final TransformationModelImpl _m = MomotUtil.getRoot(state.getModel().getEGraph(),
       * TransformationModelImpl.class);
       * System.out.println("Explore:");
       * System.out.println(_m.getTransformationchain().getUses());
       * }
       */
      // indices.add(state.getIndex());
      // Check if we exceeded the maximum state distance:
      if(maxStateDistance >= 0 && getStateDistance(state) >= maxStateDistance) {

         // START OF STATE SPACE LOCK
         synchronized(stateSpaceLock) {
            state.setOpen(false);
            state.setPruned(true);
         }
         // END OF STATE SPACE LOCK
         return Collections.emptyList();
      }

      // We need an explorer now:
      final MomotStateExplorer explorer = acquireStateExplorer();

      // Check if it is a goal state:
      if(explorer.isGoalState(state)) {

         // START OF STATE SPACE LOCK
         synchronized(stateSpaceLock) {
            state.setOpen(false);
            state.setPruned(true);
            state.setGoal(true);
         }
         // END OF STATE SPACE LOCK
         return Collections.emptyList();
      }

      // For debugging purposes:
      // checkEngineDeterminism(state);

      // Explore the state without changing the state space.
      // This can take some time. So no lock here.
      final ExplorationStepResponse r = explorer.doExplore(state);
      /*
       * System.out.println("transitions: ");
       * for(final List<List<ITransformationVariable>> t : r.transitionsTransformationStates) {
       * System.out.println(t);
       * }
       */

      final List<Transition> transitions = r.transitions;

      if(transitions.isEmpty()) {
         releaseExplorer(explorer);

         // START OF STATE SPACE LOCK
         synchronized(stateSpaceLock) {
            // Mark the state as closed:
            state.setOpen(false);
         }
         // END OF STATE SPACE LOCK
         return Collections.emptyList();
      }

      // Initialize the result.
      int newStates = 0;
      final List<State> result = new ArrayList<>(transitions.size());

      // Now check which states or transitions need to be added.
      final int count = transitions.size();
      for(int i = 0; i < count; i++) {

         // Transition details:
         final Transition t = transitions.get(i);
         final Rule rule = t.getRule();
         final int match = t.getMatch();
         final int[] parameters = t.getParameterKeys();

         // Get the hash and model of the new target state:
         final int hashCode = t.getTarget().getHashCode();
         final Model transformed = t.getTarget().getModel();

         // New state? Generate location?
         boolean newState = false;
         final int[] location = generateLocation ? shiftedLocation(state, newStates++) : null;

         // Try to find the target state without locking the state space:
         State target;
         try {
            target = getState(transformed, hashCode);
         } catch(final Throwable e) {
            target = null;
         }

         // START OF STATE SPACE LOCK
         synchronized(stateSpaceLock) {

            // Check if an equivalent state has been added in the meantime.
            if(target == null) {
               target = getState(transformed, hashCode);
            }
            // Check if the found state has been removed in the meantime.
            else if(target.getStateSpace() == null) {
               target = null;
            }

            // Ok, now we can create a new state if necessary.
            if(target == null) {
               target = createOpenState(transformed, hashCode, state, location);
               newState = true;
            }

            // Find or create the transition.
            final int m = ignoreDuplicateTransitions ? -1 : match;
            if(newState || state.getOutgoing(target, rule, m, parameters) == null) {
               t.setSource(state);
               t.setTarget(target);
               getStateSpace().incTransitionCount();
               if(stateDistanceMonitor != null) {
                  stateDistanceMonitor.updateDistance(target);
               }
            }

            // Now that the transition is there, we can decide whether to cache the model.
            if(newState) {
               addToCache(target, transformed);
               result.add(target);

               stateToTrafoVars.put(target.getHashCode(), r.transitionsTransformationStates.get(i));
               stateToRuleApps.putIfAbsent(target.getHashCode(), new ArrayList<>());
               // addNewTransformationsFromSourceState(state.getIndex(), target);
            }
            if(i == count - 1) {
               // Mark the state as closed:
               state.setOpen(false);
            }
         }
         // END OF STATE SPACE LOCK

      }

      // Release the explorer again (not earlier).
      releaseExplorer(explorer);

      // Done: return the new states.
      return result;

   }

   public Population getPopulation() {
      return this.population;
   }

   // /*
   // * Decide whether a state is open.
   // */
   // @Override
   // protected boolean isOpen(final State state) throws StateSpaceException {
   //
   // // Do a dry exploration of the state:
   // final MomotStateExplorer explorer = acquireStateExplorer();
   // final ExplorationStepResponse r = explorer.doExplore(state);
   // final List<Transition> transitions = r.transitions;
   //
   // final Set<Transition> matched = new HashSet<>();
   //
   // for(final Transition current : transitions) {
   //
   // // Find the corresponding target state in the state space.
   // final State generated = current.getTarget();
   // final State target = getState(generated.getModel(), generated.getHashCode());
   // if(target == null) {
   // releaseExplorer(explorer);
   // return true;
   // }
   //
   // // Find the corresponding outgoing transition:
   // final Transition transition = state.getOutgoing(target, current.getRule(), current.getMatch(),
   // current.getParameterKeys());
   // if(transition == null) {
   // releaseExplorer(explorer);
   // return true;
   // }
   // matched.add(transition);
   //
   // }
   // releaseExplorer(explorer);
   //
   // // Check if there are extra transitions (illegal):
   // if(!matched.containsAll(state.getOutgoing())) {
   // throw new StateSpaceException("Illegal transition in state " + state.getIndex());
   // }
   //
   // // State is not open:
   // return false;
   //
   // }

   public Map<Integer, List<List<ITransformationVariable>>> getStateToTrafoVars() {
      return this.stateToTrafoVars;
   }

   /*
    * Release a state explorer again.
    */
   protected void releaseExplorer(final MomotStateExplorer explorer) {
      synchronized(explorers) {
         explorers.push(explorer);
      }
   }

   public void setSolutionReprFunction(final MyInterface f) {
      this.solutionReprFunction = f;
   }

}
