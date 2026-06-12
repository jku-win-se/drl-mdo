package sprintplanning

import org.eclipse.emf.ecore.EObject
import org.eclipse.emf.common.util.EList
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation

import SprintPlanning.Plan

class MinimiseSprintEffortDeviation {
	
//	static def computeFitnessNormalized(Plan solution) {
//	val sprints = solution.getFeature("sprints") as EList<EObject>
//	val numSprints = solution.getFeature("maxSprints") as Integer
//
//	val fitnessValues = sprints.map [ sprint |
//		new Double((sprint.getFeature("committedItem") as EList<EObject>).fold(0d) [ result, item |
//			result + item.getFeature("Effort") as Integer
//		])
//	].toList
//
//	val effortStdDev = new StandardDeviation().evaluate(fitnessValues)
//
//	val totalEffort = fitnessValues.fold(0d) [ sum, v | sum + v ]
//	
//	   val worstCaseValues = newDoubleArrayOfSize(numSprints)
//    worstCaseValues.set(0, totalEffort)
//    // Remaining elements are already 0 by default
//    
//    val worstCaseStdDev = new StandardDeviation().evaluate(worstCaseValues)
//    
//    val normalizedFitness = effortStdDev / (if (worstCaseStdDev == 0) 1 else worstCaseStdDev)
//    
////	val normalizedFitness = effortStdDev / (if (worstCaseStdDev == 0) 1 else worstCaseStdDev)
//
//	return Math.min(1.0, Math.max(0.0, normalizedFitness))
////	return effortStdDev
//}
//	
	
	static def computeNormalizedEffortFitness(Plan solution) {

    val stdDevCalculator = new StandardDeviation

    // --- Step 1: Compute actual sprint effort values ---
    val sprintEfforts = (solution.getFeature("sprints") as EList<EObject>).map [ sprint |
        // Sum Effort over committed items; 0.0 for empty sprint
        (sprint.getFeature("committedItem") as EList<EObject>).fold(0d) [ result, item |
            result + (item.getFeature("Effort") as Integer)
        ]
    ].toList

    val actualDeviation = stdDevCalculator.evaluate(sprintEfforts)

    // --- Step 2: Compute worst-case deviation ---
    val totalEffort = sprintEfforts.reduce[p1, p2 | p1 + p2]  // sum of all sprint efforts
    val numSprints = sprintEfforts.size

    // Worst-case sprint effort standard deviation:
    // All effort in one sprint, others zero
    val worstCaseDeviation =
        if (numSprints > 0)
            totalEffort * Math.sqrt((numSprints - 1.0) / numSprints)
        else
            0.0

    // --- Step 3: Normalize (minimization: 0 → best, 1 → worst) ---
    if (worstCaseDeviation == 0.0) {
        return 0.0 // No variation possible → minimal deviation
    }

    val normalized = actualDeviation / worstCaseDeviation

    // Clamp to [0,1]
    return Math.min(1.0, Math.max(0.0, normalized))
}
	
	
	static def computeFitness(Plan solution) {
		var fitness = (solution.getFeature("sprints") as EList<EObject>).map[ sprint | 
			
			new Double((sprint.getFeature("committedItem") as EList<EObject>).fold(0d)[ result, item | // For empty sprint double is 0
				
				result + item.getFeature("Effort") as Integer
				
			])
			
		].toList // List containing 0 for empty sprints.
		
		var effortStandardDeviation = new StandardDeviation().evaluate(fitness) //Empty sprints increase deviation.
		
		//println("Sprint effort distribution: " + fitness)
		//println("Sprint effort standard deviation: " + effortStandardDeviation)
		
		return effortStandardDeviation
	}
	
	/**
	 * Helper function getting the value of the named feature (if it exists) for the given EObject.
	 */
	static def Object getFeature (EObject o, String feature) {
		
		if(o === null){
			println("Null object given")
		}
		
		o.eGet(o.eClass.getEStructuralFeature(feature))
		
	}	
}
