package at.ac.tuwien.big.momot.space.exploration;

import java.time.Duration;
import java.time.Instant;

import org.eclipse.core.runtime.IProgressMonitor;

public class ExplorationMonitor implements IProgressMonitor {

   protected Instant startTime;
   protected double elapsedTime;

   @Override
   public void beginTask(final String name, final int totalWork) {
      startTime = Instant.now();

   }

   @Override
   public void done() {
      elapsedTime = Duration.between(startTime, Instant.now()).toMillis() / 1000.0;

      System.out.println(String.format("Full exploration took %.3s seconds", this.elapsedTime));
   }

   public double getElapsedTime() {
      return this.elapsedTime;
   }

   public void info(final String preTimeMessage, final String postTimeMessage) {
      System.out.println(String.format("%s (%.3fs): %s", preTimeMessage,
            Duration.between(startTime, Instant.now()).toMillis() / 1000.0, postTimeMessage));
   }

   @Override
   public void internalWorked(final double work) {
      // TODO Auto-generated method stub

   }

   @Override
   public boolean isCanceled() {
      // TODO Auto-generated method stub
      return false;
   }

   @Override
   public void setCanceled(final boolean value) {
      // TODO Auto-generated method stub

   }

   @Override
   public void setTaskName(final String name) {
      // TODO Auto-generated method stub

   }

   @Override
   public void subTask(final String name) {
      // TODO Auto-generated method stub

   }

   @Override
   public void worked(final int work) {
      // System.out.format("Explored %d states.\n", work);

   }

}
