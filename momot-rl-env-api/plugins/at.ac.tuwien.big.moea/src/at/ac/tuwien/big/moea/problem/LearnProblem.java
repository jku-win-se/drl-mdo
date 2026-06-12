// package at.ac.tuwien.big.moea.problem;
//
// import at.ac.tuwien.big.moea.search.fitness.IFitnessFunction;
// import at.ac.tuwien.big.moea.search.solution.generator.solution.ISolutionGenerator;
//
// import org.moeaframework.core.Solution;
// import org.moeaframework.problem.AbstractProblem;
//
// public class LearnProblem<S extends Solution> extends AbstractProblem implements ISearchProblem<S> {
//
// private final ISolutionGenerator<S> solutionGenerator;
// private final IFitnessFunction<S> fitnessFunction;
//
// public LearnProblem(final IFitnessFunction<S> fitnessFunction, final ISolutionGenerator<S> solutionGenerator) {
// super(solutionGenerator.getSolutionLength(), solutionGenerator.getNrObjectives(),
// solutionGenerator.getNrConstraints());
// this.fitnessFunction = fitnessFunction;
// this.solutionGenerator = solutionGenerator;
// }
//
// @Override
// public void evaluate(final Solution solution) {
// getFitnessFunction().doEvaluate(solution, true);
// }
//
// @Override
// public IFitnessFunction<S> getFitnessFunction() {
// return fitnessFunction;
// }
//
// @Override
// public ISolutionGenerator<S> getSolutionGenerator() {
// return solutionGenerator;
// }
//
// @Override
// public Solution newSolution() {
// return getSolutionGenerator().createNewSolution();
// }
//
// }
