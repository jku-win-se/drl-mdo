package at.ac.tuwien.big.moea.search.algorithm.reinforcement.datastructures;

import java.util.List;

public class DeltaUpdate {
   public final List<IApplicationState> state;
   public final List<IApplicationState> action;
   public final double oldValue;
   public final boolean yieldsNewState;

   public DeltaUpdate(final List<IApplicationState> state, final List<IApplicationState> action, final double oldValue,
         final boolean yieldsNewState) {
      this.state = state;
      this.action = action;
      this.oldValue = oldValue;
      this.yieldsNewState = yieldsNewState;
   }
}
