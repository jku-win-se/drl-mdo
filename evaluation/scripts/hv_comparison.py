#!/usr/bin/env python3

import sys
import ast
from itertools import combinations
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from scipy.stats import mannwhitneyu
import paretoset
from pymoo.indicators.hv import HV
from statsmodels.stats.multitest import multipletests


SIGNIFICANCE_LEVEL = 0.01
EXCLUDED_TEST_PAIRS = [
    ("NSGAII-2", "Q+"),
]

# Set order for effect size: PREFERRED_ALGO_A will be compared against PREFERRED_ALGO_B
PREFERRED_ALGO_A = {"100k", "conv"}
PREFERRED_ALGO_B = {"NSGAII-2", "Q+"}


# Whitelist of algorithm folder names (keys) and display labels (values) for tables/output.
algo_mapping = {
"Q+": "Q+",
    "NSGAII-2": "NSGA-II",
    "100k-10": "ARTC_100k-10",
    "100k": "ARTC_100k",
    "QLearningExplore": "Q+",
    "best-train-policy-10": "best-train-policy-10",
    "best-train-policy-converge": "best-train-policy-converge",
        "best-pre100-policy-conv": "best-pre100-policy-conv",
    "best-policy-10": "best-policy-10",
    "best-policy-conv": "best-policy-conv",
    "os": "OS",
    "10": "best-policy-10",
    "conv": "best-policy-conv",
}


def parse_solution_file(file_path: Path, remove_zero_dims: int=None) -> Dict[int, List[List[float]]]:
    """
    Parse nn-solutions.txt file and extract solutions at each iteration.
    
    Returns:
        Dictionary mapping iteration numbers to lists of solution vectors
    """
    solutions_by_iteration = {}
    current_iteration = None
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check if line is an iteration number
                try:
                    iteration_num = int(line)
                    current_iteration = iteration_num
                    solutions_by_iteration[current_iteration] = []
                except ValueError:
                    # Line is not an iteration number, should be a solution vector
                    if current_iteration is None:
                        current_iteration = 9999
                        solutions_by_iteration[current_iteration] = []
                    try:
                        # Parse solution vector: "[1.3, 4]" -> [1.3, 4.0]
                        solution = ast.literal_eval(line)
                        if isinstance(solution, list):
                            if remove_zero_dims is None or float(solution[int(remove_zero_dims)]) != .0:
                                solutions_by_iteration[current_iteration].append(solution)
                    except (ValueError, SyntaxError) as e:
                        print(f"Warning: Could not parse solution vector '{line}' in {file_path}: {e}")
                        continue
    
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return {}
    
    return solutions_by_iteration


def load_all_runs(algorithm_path: Path, remove_zero_dims: int=None) -> Dict[str, Dict[int, List[List[float]]]]:
    """
    Load solutions from all run subfolders for a single algorithm.
    
    Returns:
        Dictionary mapping run names to their solutions by iteration
    """
    all_runs = {}
    
    if not algorithm_path.exists():
        print(f"Error: Algorithm path {algorithm_path} does not exist")
        return {}
    
    # Find all subdirectories (runs)
    run_dirs = [d for d in algorithm_path.iterdir() if d.is_dir()]
    
    if not run_dirs:
        print(f"Warning: No subdirectories found in {algorithm_path}")
        return {}
    
    print(f"Found {len(run_dirs)} run directories in {algorithm_path.name}")
    
    for run_dir in run_dirs:
        ## Each run directory should contain either nn-solutions.txt or soltuions_final.txt
        print(f"Processing run directory: {run_dir.name}")
        # Get solution file which is either nn-solutions.txt or solutions_final.txt
        solution_file = run_dir / "solutions_final.txt"
        if not solution_file.exists():
            solution_file = run_dir / "nn-solutions.txt"
        if not solution_file.exists():
            solution_file = run_dir / "nn-solutions_final.txt"
        if not solution_file.exists():
            print(f"Warning: No solution file found in {run_dir}, skipping this run.")
            continue
        if solution_file.exists():
            #print(f"Loading run: {run_dir.name}")
            solutions = parse_solution_file(solution_file, remove_zero_dims)
            if solutions:
                all_runs[run_dir.name] = solutions
            else:
                print(f"Warning: No solutions found in {run_dir.name}")
        else:
            print(f"Warning: nn-solutions.txt not found in {run_dir}")
    
    return all_runs


def load_multiple_algorithms(algorithm_paths: List[Path], remove_zero_dims: int=None) -> Dict[str, Dict[str, Dict[int, List[List[float]]]]]:
    """
    Load solutions from multiple algorithm directories.
    
    Returns:
        Dictionary mapping algorithm names to their runs data
    """
    all_algorithms = {}
    
    for algo_path in algorithm_paths:
        algorithm_name = algo_path.name
        print(f"\nLoading algorithm: {algorithm_name}")
        runs_data = load_all_runs(algo_path, remove_zero_dims)

        if runs_data:
            all_algorithms[algorithm_name] = runs_data
            print(f"Successfully loaded {len(runs_data)} runs for {algorithm_name}")
        else:
            print(f"Warning: No valid runs found for {algorithm_name}")
    
    return all_algorithms


def convert_to_minimization(solutions: List[List[float]], minimization_flags: List[bool]) -> np.ndarray:
    """
    Convert solutions to minimization form based on minimization flags.
    For maximization objectives, multiply by -1.
    """
    solutions_array = np.array(solutions)
    for i, is_min in enumerate(minimization_flags):
        if not is_min:  # If maximization objective
            solutions_array[:, i] *= -1
    return solutions_array


def build_combined_reference_set(all_algorithms: Dict[str, Dict[str, Dict[int, List[List[float]]]]], 
                               minimization_flags: List[bool]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build combined reference Pareto set from final iteration solutions of all algorithms and runs.
    Also returns the ideal (minimum) and nadir (maximum) points for normalization.
    
    Returns:
        Tuple of (reference_set, ideal_point, nadir_point)
    """
    all_final_solutions = []
    
    for algorithm_name, algorithm_runs in all_algorithms.items():
        print(f"Collecting final solutions from {algorithm_name}")
        
        for run_name, run_data in algorithm_runs.items():
            if not run_data:
                continue
                
            # Get solutions from the last iteration
            max_iteration = max(run_data.keys())
            final_solutions = run_data[max_iteration]
            
            if final_solutions:
                # Convert to minimization form
                converted_solutions = convert_to_minimization(final_solutions, minimization_flags)
                all_final_solutions.extend(converted_solutions.tolist())
    
    if not all_final_solutions:
        raise ValueError("No final solutions found across all algorithms and runs")
    
    # Find Pareto-optimal solutions from all algorithms combined
    all_final_array = np.array(all_final_solutions)
    pareto_mask = paretoset.paretoset(all_final_array, sense=["min"] * len(minimization_flags))
    reference_set = all_final_array[pareto_mask]
    
    # Calculate ideal point (minimum across all objectives) and nadir point (maximum)
    # Using ALL solutions, not just the Pareto front, to get proper bounds
    ideal_point = all_final_array.min(axis=0)
    nadir_point = all_final_array.max(axis=0)
    
    print(f"Combined reference Pareto set contains {len(reference_set)} solutions from {len(all_final_solutions)} total solutions")
    print(f"Ideal point (min values): {ideal_point}")
    print(f"Nadir point (max values): {nadir_point}")
    
    return reference_set, ideal_point, nadir_point


def normalize_solutions(solutions: np.ndarray, ideal_point: np.ndarray, nadir_point: np.ndarray) -> np.ndarray:
    """
    Normalize solutions to [0, 1] range using ideal and nadir points.
    This matches MOEA Framework's normalization approach.
    
    Args:
        solutions: Array of solutions in minimization form
        ideal_point: Minimum values for each objective
        nadir_point: Maximum values for each objective
        
    Returns:
        Normalized solutions in [0, 1] range
    """
    # Avoid division by zero
    ranges = nadir_point - ideal_point
    ranges = np.where(ranges == 0, 1.0, ranges)  # Replace zero ranges with 1 to avoid division by zero
    
    normalized = (solutions - ideal_point) / ranges
    
    # Clip to [0, 1] to handle any numerical issues or solutions outside bounds
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized


def calculate_reference_point(reference_set: np.ndarray, minimization_flags: List[bool], 
                            ideal_point: np.ndarray = None, nadir_point: np.ndarray = None) -> np.ndarray:
    """
    Calculate reference point for hypervolume calculation.
    
    When using normalized solutions (MOEA Framework style), returns (1.0, 1.0, ...) 
    plus a small offset to ensure all Pareto optimal solutions contribute to HV.
    
    For unnormalized solutions, uses maximum values with a margin.
    """
    if ideal_point is not None and nadir_point is not None:
        # MOEA Framework style: use (1 + epsilon) for normalized solutions
        # The small offset ensures solutions on the boundary still contribute to HV
        n_objectives = len(minimization_flags)
        reference_point = np.ones(n_objectives) * 1.0  # Use 1.0 for normalized space
        print(f"Using normalized reference point: {reference_point}")
    else:
        # Fallback: For minimization problems, reference point should be worse (higher) than all solutions
        margins = np.abs(reference_set.max(axis=0) - reference_set.min(axis=0)) * 0.1
        margins = np.maximum(margins, 1.0)  # Ensure minimum margin
        reference_point = reference_set.max(axis=0) + margins
        print(f"Calculated reference point: {reference_point}")
    
    return reference_point


def filter_dominated_solutions(solutions: np.ndarray, minimization_flags: List[bool]) -> np.ndarray:
    """
    Filter out dominated solutions, keeping only non-dominated (Pareto optimal) solutions.
    This matches MOEA Framework's behavior of only considering non-dominated solutions for HV.
    
    Args:
        solutions: Array of solutions in minimization form
        minimization_flags: List indicating which objectives are minimized
        
    Returns:
        Array of non-dominated solutions
    """
    if len(solutions) == 0:
        return solutions
    
    pareto_mask = paretoset.paretoset(solutions, sense=["min"] * len(minimization_flags))
    return solutions[pareto_mask]


def calculate_algorithm_hypervolumes(all_algorithms: Dict[str, Dict[str, Dict[int, List[List[float]]]]], 
                                   reference_point: np.ndarray,
                                   minimization_flags: List[bool],
                                   ideal_point: np.ndarray = None,
                                   nadir_point: np.ndarray = None) -> Dict[str, List[float]]:
    """
    Calculate hypervolume for each algorithm's runs using MOEA Framework compatible approach.
    
    This implementation:
    1. Normalizes all solutions using the combined reference set bounds
    2. Filters to only non-dominated solutions per run
    3. Calculates HV using a consistent reference point
    
    Args:
        all_algorithms: Dictionary mapping algorithm names to their runs data
        reference_point: Reference point for HV calculation (in normalized space if using normalization)
        minimization_flags: List indicating which objectives are minimized
        ideal_point: Minimum values for normalization (optional)
        nadir_point: Maximum values for normalization (optional)
        
    Returns:
        Dictionary mapping algorithm names to lists of hypervolumes (one per run)
    """
    algorithm_hypervolumes = {}
    use_normalization = ideal_point is not None and nadir_point is not None
    
    # Create HV indicator with the reference point
    hv_indicator = HV(ref_point=reference_point)
    
    for algorithm_name, algorithm_runs in all_algorithms.items():
        print(f"Calculating hypervolumes for {algorithm_name}")
        hypervolumes = []
        
        for run_name, run_data in algorithm_runs.items():
            if not run_data:
                continue
            
            # Get solutions from the last iteration (final Pareto front)
            max_iteration = max(run_data.keys())
            final_solutions = run_data[max_iteration]
            
            if final_solutions:
                # Convert to minimization form
                solutions = convert_to_minimization(final_solutions, minimization_flags)
                
                # Filter to non-dominated solutions only (MOEA Framework behavior)
                solutions = filter_dominated_solutions(solutions, minimization_flags)
                
                if len(solutions) == 0:
                    print(f"  Warning: No non-dominated solutions in {algorithm_name} run {run_name}")
                    continue
                
                # Normalize solutions if using normalization (MOEA Framework style)
                if use_normalization:
                    solutions = normalize_solutions(solutions, ideal_point, nadir_point)
                
                try:
                    # Calculate hypervolume
                    hv_value = hv_indicator(solutions)
                    hypervolumes.append(hv_value)
                    print(f"  Run {run_name}: HV = {hv_value:.6f} ({len(solutions)} non-dominated solutions)")
                except Exception as e:
                    print(f"Warning: Could not calculate HV for {algorithm_name} run {run_name}: {e}")
        
        if hypervolumes:
            algorithm_hypervolumes[algorithm_name] = hypervolumes
            mean_hv = np.mean(hypervolumes)
            std_hv = np.std(hypervolumes)
            print(f"  {algorithm_name} - Median HV: {np.median(hypervolumes):.6f}, Mean HV: {mean_hv:.6f} ± {std_hv:.6f} ({len(hypervolumes)} runs)")
        else:
            print(f"Warning: No valid hypervolumes calculated for {algorithm_name}")
    
    return algorithm_hypervolumes


def calculate_algorithm_pareto_sets(all_algorithms: Dict[str, Dict[str, Dict[int, List[List[float]]]]], 
                                  minimization_flags: List[bool]) -> Dict[str, np.ndarray]:
    """
    Calculate Pareto set for each algorithm across all its runs.
    
    Returns:
        Dictionary mapping algorithm names to their Pareto sets
    """
    algorithm_pareto_sets = {}
    
    for algorithm_name, algorithm_runs in all_algorithms.items():
        print(f"Calculating Pareto set for {algorithm_name}")
        all_algorithm_solutions = []
        
        # Collect all final solutions from all runs of this algorithm
        for run_name, run_data in algorithm_runs.items():
            if not run_data:
                continue
                
            # Get solutions from the last iteration
            max_iteration = max(run_data.keys())
            final_solutions = run_data[max_iteration]
            
            if final_solutions:
                # Convert to minimization form
                converted_solutions = convert_to_minimization(final_solutions, minimization_flags)
                all_algorithm_solutions.extend(converted_solutions.tolist())
        
        if all_algorithm_solutions:
            # Find Pareto-optimal solutions for this algorithm
            algorithm_array = np.array(all_algorithm_solutions)
            pareto_mask = paretoset.paretoset(algorithm_array, sense=["min"] * len(minimization_flags))
            algorithm_pareto_set = algorithm_array[pareto_mask]
            algorithm_pareto_sets[algorithm_name] = algorithm_pareto_set
            print(f"  {algorithm_name} Pareto set contains {len(algorithm_pareto_set)} solutions from {len(all_algorithm_solutions)} total solutions")
        else:
            print(f"Warning: No solutions found for {algorithm_name}")
    
    return algorithm_pareto_sets


def save_results_to_text(algorithm_pareto_sets: Dict[str, np.ndarray],
                        algorithm_hypervolumes: Dict[str, List[float]],
                        reference_set: np.ndarray,
                        minimization_flags: List[bool],
                        output_dir: Path):
    """
    Save comprehensive results to a text file.
    """
    output_file = output_dir / "algorithm_comparison_results.txt"
    
    with open(output_file, 'w') as f:
        f.write("Multi-Objective Algorithm Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Write minimization flags
        f.write(f"Minimization flags: {minimization_flags}\n")
        f.write("(True = minimization, False = maximization)\n\n")
        
        # Write reference set
        f.write(f"Combined Reference Pareto Set ({len(reference_set)} solutions):\n")
        f.write("-" * 40 + "\n")
        for i, sol in enumerate(reference_set):
            f.write(f"  {i+1:3d}: {sol}\n")
        f.write("\n")
        
        # Write results for each algorithm
        for algorithm_name in algorithm_pareto_sets.keys():
            f.write(f"Algorithm: {algorithm_name}\n")
            f.write("-" * 30 + "\n")
            
            # Hypervolume statistics
            if algorithm_name in algorithm_hypervolumes:
                hvs = algorithm_hypervolumes[algorithm_name]
                mean_hv = np.mean(hvs)
                std_hv = np.std(hvs)
                min_hv = np.min(hvs)
                max_hv = np.max(hvs)
                f.write(f"Hypervolume Statistics:\n")
                f.write(f"  Median HV: {np.median(hvs):.6f}\n")
                f.write(f"  Mean HV: {mean_hv:.6f}\n")
                f.write(f"  Std HV:  {std_hv:.6f}\n")
                f.write(f"  Min HV:  {min_hv:.6f}\n")
                f.write(f"  Max HV:  {max_hv:.6f}\n")
                f.write(f"  Runs:    {len(hvs)}\n")
                f.write(f"  Individual HVs: {[f'{hv:.6f}' for hv in hvs]}\n\n")
            
            # Pareto set
            pareto_set = algorithm_pareto_sets[algorithm_name]
            f.write(f"Pareto Set ({len(pareto_set)} solutions):\n")
            for i, sol in enumerate(pareto_set):
                f.write(f"  {i+1:3d}: {sol}\n")
            f.write("\n\n")
    
    print(f"Detailed results saved to: {output_file}")


def save_hypervolume_data(algorithm_hypervolumes: Dict[str, List[float]],
                          output_dir: Path) -> None:
    """Save per-run hypervolume values to CSV."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data = []
    for algorithm, hvs in algorithm_hypervolumes.items():
        for i, hv in enumerate(hvs):
            csv_data.append({
                'Algorithm': algorithm,
                'Run': i + 1,
                'Hypervolume': hv
            })

    df = pd.DataFrame(csv_data)
    csv_path = output_dir / "hypervolume_comparison_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Hypervolume data saved to: {csv_path}")


def vargha_delaney_a12(a: List[float], b: List[float]) -> float:
    """
    Compute Vargha-Delaney A12 effect size.

    A12 = (#{(x,y): x > y} + 0.5 * #{(x,y): x == y}) / (m * n)

    A12 > 0.5 means the first distribution (a) stochastically dominates the second (b).
    A12 = 0.5 indicates no difference.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.5
    greater = sum(1 for x in a for y in b if x > y)
    equal   = sum(1 for x in a for y in b if x == y)
    return (greater + 0.5 * equal) / (m * n)


def effect_size_label(a12: float) -> str:
    """
    Classify the Vargha-Delaney A12 effect size using the thresholds from
    Vargha & Delaney (2000):
        |A12 - 0.5| < 0.06  -> 'negligible'
        |A12 - 0.5| < 0.14  -> 'small'
        |A12 - 0.5| < 0.21  -> 'medium'
        otherwise           -> 'large'
    """
    d = abs(a12 - 0.5)
    if d < 0.056:
        return "negligible"
    elif d < 0.138  :
        return "small"
    elif d < 0.214:
        return "medium"
    else:
        return "large"


def should_skip_statistical_test(algo_a: str, algo_b: str) -> bool:
    """Return True when a pair should be excluded from statistical testing."""
    identifiers_a = {algo_a, algo_mapping.get(algo_a, algo_a)}
    identifiers_b = {algo_b, algo_mapping.get(algo_b, algo_b)}

    for excluded_a, excluded_b in EXCLUDED_TEST_PAIRS:
        if excluded_a in identifiers_a and excluded_b in identifiers_b:
            return True
        if excluded_b in identifiers_a and excluded_a in identifiers_b:
            return True

    return False


def orient_statistical_test_pair(algo_a: str, algo_b: str) -> Tuple[str, str]:
    """Orient a pair so preferred algorithms appear on the requested side."""
    identifiers_a = {algo_a, algo_mapping.get(algo_a, algo_a)}
    identifiers_b = {algo_b, algo_mapping.get(algo_b, algo_b)}

    prefers_a_as_a = any(identifier in PREFERRED_ALGO_A for identifier in identifiers_a)
    prefers_b_as_a = any(identifier in PREFERRED_ALGO_A for identifier in identifiers_b)
    prefers_a_as_b = any(identifier in PREFERRED_ALGO_B for identifier in identifiers_a)
    prefers_b_as_b = any(identifier in PREFERRED_ALGO_B for identifier in identifiers_b)

    if prefers_a_as_a and not prefers_b_as_a:
        return algo_a, algo_b

    if prefers_b_as_a and not prefers_a_as_a:
        return algo_b, algo_a

    if prefers_a_as_b and not prefers_b_as_b:
        return algo_b, algo_a

    return algo_a, algo_b


def calculate_statistical_tests(
    algorithm_hypervolumes: Dict[str, List[float]]
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Perform pairwise Mann-Whitney U tests and compute Vargha-Delaney A12 effect
    sizes for all algorithm pairs based on their hypervolume distributions.

    P-values are adjusted with the Holm correction across all unique pairwise
    algorithm comparisons for the current instance.

    Returns:
        Dictionary keyed by (algo_A, algo_B) with:
            - 'statistic': Mann-Whitney U statistic
            - 'raw_p_value': two-sided raw p-value
            - 'p_value'  : Holm-adjusted p-value for the full instance-wide family
            - 'a12'      : Vargha-Delaney A12 (A12 > 0.5 => algo_A tends to have higher HV)
            - 'effect'   : magnitude label (negligible / small / medium / large)
    """
    algo_names = list(algorithm_hypervolumes.keys())
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    pair_results = []
    raw_p_values = []

    print("\n" + "=" * 110)
    print("Pairwise Statistical Tests (hypervolume)")
    print("=" * 110)
    header = (
        f"{'Algo A':<25} {'Algo B':<25} {'U-stat':>10} {'raw p':>10} "
        f"{'Holm p':>10} {'A12':>6} {'Effect':<12} {'Sig.(adj<0.01)'}"
    )
    print(header)
    print("-" * len(header))

    for algo_a, algo_b in combinations(algo_names, 2):
        if should_skip_statistical_test(algo_a, algo_b):
            print(f"  Skipping ({algo_a}, {algo_b}): excluded from statistical testing.")
            continue

        algo_a, algo_b = orient_statistical_test_pair(algo_a, algo_b)

        hvs_a = algorithm_hypervolumes[algo_a]
        hvs_b = algorithm_hypervolumes[algo_b]

        if len(hvs_a) < 2 or len(hvs_b) < 2:
            print(f"  Skipping ({algo_a}, {algo_b}): need at least 2 runs per algorithm.")
            continue

        stat, raw_p_value = mannwhitneyu(hvs_a, hvs_b, alternative='two-sided')
        a12 = vargha_delaney_a12(hvs_a, hvs_b)
        label = effect_size_label(a12)

        pair_results.append((algo_a, algo_b, stat, raw_p_value, a12, label))
        raw_p_values.append(raw_p_value)

    if not pair_results:
        print("=" * 110)
        print("Note: A12 > 0.5 means Algo A tends to achieve higher hypervolume than Algo B.")
        print("Holm correction is applied across all pairwise comparisons in this instance.")
        print("=" * 110 + "\n")
        return results

    rejected, adjusted_p_values, _, _ = multipletests(
        raw_p_values,
        alpha=SIGNIFICANCE_LEVEL,
        method='holm',
    )
    family_size = len(raw_p_values) 

    for (algo_a, algo_b, stat, raw_p_value, a12, label), adjusted_p_value, reject in zip(
        pair_results, adjusted_p_values, rejected
    ):
        significant = "Yes" if reject else "No"
        results[(algo_a, algo_b)] = {
            'statistic': stat,
            'raw_p_value': raw_p_value,
            'p_value': adjusted_p_value,
            'a12': a12,
            'effect': label,
            'significant': bool(reject),
            'correction_method': 'holm',
            'family_size': family_size,
        }

        a_display = algo_a[:24]
        b_display = algo_b[:24]
        print(
            f"{a_display:<25} {b_display:<25} {stat:>10.1f} "
            f"{raw_p_value:>10.4f} {adjusted_p_value:>10.4f} "
            f"{a12:>6.4f} {label:<12} {significant}"
        )

    print("=" * 110)
    print("Note: A12 > 0.5 means Algo A tends to achieve higher hypervolume than Algo B.")
    print("Holm correction is applied across all pairwise comparisons in this instance.")
    print("=" * 110 + "\n")

    return results


def save_statistical_tests(
    test_results: Dict[Tuple[str, str], Dict[str, Any]],
    algorithm_hypervolumes: Dict[str, List[float]],
    output_dir: Path
) -> None:
    """
    Save pairwise statistical test results to a human-readable text file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Text file ----
    txt_path = output_dir / "statistical_tests.txt"
    with open(txt_path, 'w') as f:
        f.write("Pairwise Statistical Tests (hypervolume)\n")
        f.write("=" * 110 + "\n")
        f.write("Mann-Whitney U (two-sided) + Vargha-Delaney A12 effect size\n")
        f.write("P-values are Holm-adjusted across all pairwise comparisons in this instance.\n")
        f.write("Comparisons involving 100k or conv are oriented with that algorithm as Algo A.\n")
        f.write("Comparisons involving NSGAII-2 or Q+ are oriented with that algorithm as Algo B.\n")
        if EXCLUDED_TEST_PAIRS:
            excluded_pairs = ", ".join(f"{left} vs {right}" for left, right in EXCLUDED_TEST_PAIRS)
            f.write(f"Excluded comparisons: {excluded_pairs}.\n")
        f.write("Effect size thresholds (Vargha & Delaney, 2000):\n")
        f.write("  |A12-0.5| < 0.06 -> negligible\n")
        f.write("  |A12-0.5| < 0.14 -> small\n")
        f.write("  |A12-0.5| < 0.21 -> medium\n")
        f.write("  |A12-0.5| >= 0.21 -> large\n")
        f.write("A12 > 0.5 means Algo A tends to achieve higher hypervolume.\n")
        f.write("Column 'p-value' below is the Holm-adjusted value; raw p-values are shown separately.\n")
        f.write("=" * 110 + "\n\n")

        # Per-algorithm HV summary
        f.write("Hypervolume Summary per Algorithm:\n")
        f.write("-" * 50 + "\n")
        for algo, hvs in algorithm_hypervolumes.items():
            f.write(f"  {algo}:\n")
            f.write(f"    n={len(hvs)}, median={np.median(hvs):.6f}, "
                    f"mean={np.mean(hvs):.6f}, std={np.std(hvs):.6f}, "
                    f"min={np.min(hvs):.6f}, max={np.max(hvs):.6f}\n")
        f.write("\n")

        # Pairwise table
        header = (
            f"{'Algo A':<25} {'Algo B':<25} {'U-stat':>10} {'raw p':>10} "
            f"{'p-value':>10} {'A12':>6} {'Effect':<12} {'Family':>6} {'Sig.(adj<0.01)'}\n"
        )
        f.write(header)
        f.write("-" * len(header.rstrip()) + "\n")
        for (algo_a, algo_b), res in test_results.items():
            a_display = algo_a[:24]
            b_display = algo_b[:24]
            significant = "Yes" if res['significant'] else "No"
            f.write(
                f"{a_display:<25} {b_display:<25} "
                f"{res['statistic']:>10.1f} {res['raw_p_value']:>10.4f} {res['p_value']:>10.4f} "
                f"{res['a12']:>6.4f} {res['effect']:<12} {res['family_size']:>6} {significant}\n"
            )
    print(f"Statistical test results saved to: {txt_path}")


def save_latex_hv_table(
    algorithm_hypervolumes: Dict[str, List[float]],
    output_dir: Path,
    title: str = "",
) -> None:
    """
    Write a LaTeX table snippet with mean HV & std HV per algorithm,
    rounded to 3 decimal places.

    Algorithms are ordered by their position in algo_mapping.
    Each algorithm occupies one row with an empty placeholder cell:
        Algorithm & <mean_hv> & <std_hv> & <placeholder> \\\\
    """
    # Order algorithms by their position in algo_mapping; unknowns go last
    mapping_keys = list(algo_mapping.keys())
    def _sort_key(algo: str) -> int:
        try:
            return mapping_keys.index(algo)
        except ValueError:
            return len(mapping_keys)

    ordered_algos = sorted(algorithm_hypervolumes.keys(), key=_sort_key)

    tex_path = output_dir / "hv_latex_table.tex"
    with open(tex_path, "w") as f:
        # Table preamble
        f.write("% Hypervolume comparison table\n")
        if title:
            f.write(f"% Case study / instance: {title}\n")
        f.write("%\n")
        f.write("\\begin{tabular}{lrrl}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & Mean HV & Std HV & \\\\\n")
        f.write("\\midrule\n")

        for algo in ordered_algos:
            hvs = algorithm_hypervolumes[algo]
            display_name = algo_mapping.get(algo, algo)
            mean_hv = round(float(np.mean(hvs)), 3)
            std_hv  = round(float(np.std(hvs)),  3)
            mean_str = f"{mean_hv:.3f}".lstrip('0')
            std_str  = f"{std_hv:.3f}".lstrip('0')
            f.write(f"{display_name} & {mean_str} & {std_str} & \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"LaTeX HV table saved to:         {tex_path}")


def main():
    """Main execution function."""

    script_dir = Path(__file__).resolve().parent
    algorithm_roots = [
        script_dir / "../rq2-performance/slb/SLB-50/solutions",
    ]
    minimization_flags = [True, True]  
    output_dir = script_dir / "../rq2-performance/slb/SLB-50"
    title = "SLB-50"
    remove_zero_in_dims = 1  # e.g. 0 to drop solutions with 0 in that dimension
    # ----------------------------------

    try:
        algorithm_paths = [Path(p).expanduser().resolve() for p in algorithm_roots]
        algorithm_paths = [p for p in algorithm_paths if p.is_dir()]
        algorithm_paths = [
            p for a in algorithm_paths
            for p in a.iterdir()
            if p.is_dir() and p.name in algo_mapping
        ]
    except Exception as e:
        print(f"Error resolving algorithm paths: {e}")
        sys.exit(1)

    print(f"Algorithm paths: {[str(p) for p in algorithm_paths]}")
    print(f"Minimization flags: {minimization_flags}")

    # Load all algorithms
    print("Loading experimental runs from all algorithms...")
    all_algorithms = load_multiple_algorithms(algorithm_paths, remove_zero_in_dims)

    if not all_algorithms:
        print("Error: No valid algorithms found")
        sys.exit(1)

    total_runs = sum(len(runs) for runs in all_algorithms.values())
    print(f"Successfully loaded {len(all_algorithms)} algorithms with {total_runs} total runs")

    # Build combined reference Pareto set with ideal and nadir points
    print("\nBuilding combined reference Pareto set...")
    try:
        reference_set, ideal_point, nadir_point = build_combined_reference_set(
            all_algorithms, minimization_flags
        )
    except Exception as e:
        print(f"Error building reference set: {e}")
        sys.exit(1)

    
    reference_point = calculate_reference_point(
            reference_set, minimization_flags, ideal_point, nadir_point
    )

    # Calculate hypervolumes for each algorithm (MOEA Framework compatible)
    print("\nCalculating hypervolumes for each algorithm (MOEA Framework compatible)...")
    algorithm_hypervolumes = calculate_algorithm_hypervolumes(
        all_algorithms, reference_point, minimization_flags, ideal_point, nadir_point
    )

    if not algorithm_hypervolumes:
        print("Error: No hypervolumes calculated")
        sys.exit(1)

    # Statistical significance tests (Mann-Whitney U + Vargha-Delaney A12)
    print("\nRunning pairwise statistical tests...")
    statistical_test_results = calculate_statistical_tests(algorithm_hypervolumes)

    # Calculate Pareto sets for each algorithm
    print("\nCalculating individual Pareto sets for each algorithm...")
    algorithm_pareto_sets = calculate_algorithm_pareto_sets(all_algorithms, minimization_flags)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results to text file
    print("\nSaving detailed results to text file...")
    save_results_to_text(
        algorithm_pareto_sets, algorithm_hypervolumes,
        reference_set, minimization_flags, output_dir
    )

    # Save statistical test results
    print("\nSaving statistical test results...")
    save_statistical_tests(statistical_test_results, algorithm_hypervolumes, output_dir)

    # Save LaTeX HV table
    print("\nSaving LaTeX HV table...")
    save_latex_hv_table(algorithm_hypervolumes, output_dir, title)

    # Save per-run hypervolume data
    print("\nSaving hypervolume data...")
    save_hypervolume_data(algorithm_hypervolumes, output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
