package at.ac.tuwien.big.momot.search.algorithm.reinforcement.datastructures;

import at.ac.tuwien.big.moea.search.algorithm.reinforcement.utils.IRLUtils;
import at.ac.tuwien.big.momot.problem.solution.TransformationSolution;
import at.ac.tuwien.big.momot.search.algorithm.reinforcement.algorithm.RLUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import org.eclipse.emf.henshin.interpreter.Engine;
import org.moeaframework.core.Solution;

public class SolutionArchive<S extends Solution> {

   private static class Cell<S extends Solution> {

      final int step;
      double value;
      S solution;
      int nSelections = 0;

      Cell(final S solution, final int step, final double value) {
         this.step = step;
         this.solution = solution;
         this.value = value;
      }

      @Override
      public String toString() {
         return String.format("Step=%d, Value=%.3f, nSel=%d", step, value, nSelections);
      }

      double weight() {
         return 1.0 / (Math.sqrt(nSelections) + 1.0);
      }
   }

   private final Map<Integer, Cell<S>> byStep = new HashMap<>();
   private final IRLUtils<S> rlUtils = new RLUtils();
   private final List<Cell<S>> allSolutions = new ArrayList<>();

   public synchronized void addOrUpdate(final S solution, final int step, final double value) {
      final Cell<S> existing = byStep.get(step);
      if(existing == null) {
         final Cell<S> s = new Cell<>(solution, step, value);
         byStep.put(step, s);
         allSolutions.add(s);
      } else if(value > existing.value) {
         existing.value = value;
      }
   }

   public synchronized List<String> debugList() {
      final List<String> lines = new ArrayList<>();
      for(final Cell<S> s : allSolutions) {
         lines.add(s.toString());
      }
      return lines;
   }

   /**
    * synchronized ensures nSelections increments safely
    *
    * @param engine
    */
   public synchronized S retrieve(final Engine engine) {
      if(allSolutions.isEmpty()) {
         return null;
      }

      double totalWeight = 0.0;
      for(final Cell<S> s : allSolutions) {
         totalWeight += s.weight();
      }

      final double r = ThreadLocalRandom.current().nextDouble(totalWeight);
      double cum = 0.0;
      for(final Cell<S> s : allSolutions) {
         cum += s.weight();
         if(r <= cum) {
            s.nSelections++;

            if(engine != null) {
               final TransformationSolution ts = ((TransformationSolution) s.solution).copy(engine);
               return (S) ts;
            }
            return (S) s.solution.copy();
         }
      }

      // fallback
      final Cell<S> last = allSolutions.get(allSolutions.size() - 1);
      last.nSelections++;
      if(engine != null) {
         final TransformationSolution ts = ((TransformationSolution) last.solution).copy(engine);
         return (S) ts;
      }
      return (S) last.solution.copy();
   }

   public synchronized int size() {
      return allSolutions.size();
   }
}
