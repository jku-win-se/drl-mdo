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
package at.ac.tuwien.big.moea.experiment.executor.listener;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.time.StopWatch;
import org.moeaframework.util.progress.ProgressEvent;

public class SeedRuntimePrintListener extends AbstractProgressListener {

   private final StopWatch seedWatch = new StopWatch();
   private final StopWatch totalWatch = new StopWatch();
   private final List<Long> runtimes = new ArrayList<>();

   @Override
   public void update(final ProgressEvent event) {
      if(isStarted(event)) {
         runtimes.clear();
         if(!totalWatch.isStarted()) {
            totalWatch.start();
         }
      }

      if(isStarted(event) || isSeedStarted(event)) {
         if(!seedWatch.isStarted()) {
            seedWatch.start();
         }
         println("Run " + event.getCurrentSeed() + " of " + event.getTotalSeeds() + " started.");
      }

      if(isSeedFinished(event)) {
         seedWatch.stop();
         println("Run " + event.getCurrentSeed() + " of " + event.getTotalSeeds() + " terminated after " + seedWatch
               + " (" + seedWatch.getTime() + " ms).");
         runtimes.add(seedWatch.getTime());
         seedWatch.reset();
      }

      if(isFinished(event)) {
         double sum = 0;
         for(final Long runtime : runtimes) {
            sum += runtime;
         }
         final double meanRuntime = sum / 1000.0 / event.getTotalSeeds();
         final double meanRuntimeWOutFirst = (sum - runtimes.get(0)) / 1000.0 / (event.getTotalSeeds() - 1);
         totalWatch.stop();
         println("Total runtime for " + event.getTotalSeeds() + " seeds: " + totalWatch + " (" + totalWatch.getTime()
               + " ms). -> Mean: " + String.format("%.1f", meanRuntime) + "s " + " ("
               + String.format("%.1f", meanRuntimeWOutFirst) + " without 1st)");
         totalWatch.reset();
      }
   }

}
