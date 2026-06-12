#!/usr/bin/env python3
"""
Multi-objective Optimization Convergence Analysis Tool

This script analyzes the convergence of multi-objective optimization algorithms
by calculating Hypervolume (HV) and Inverted Generational Distance (IGD) metrics
over time, aggregated across multiple experimental runs.

Usage:
    python convergence_plotter.py <algorithm_path> <minimization_flags>
    
Example:
    python convergence_plotter.py ./nsga2_results "True,False"
    # This indicates objective 1 is minimized, objective 2 is maximized
"""

import os
import sys
import ast
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import warnings

# Required libraries (install with: pip install paretoset pymoo matplotlib numpy)
try:
    import paretoset
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install required packages:")
    print("pip install paretoset pymoo matplotlib numpy")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot HV and IGD convergence for multi-objective optimization results"
    )
    parser.add_argument(
        "algorithm_path",
        type=str,
        help="Path to the algorithm results directory containing run subfolders"
    )
    parser.add_argument(
        "minimization_flags",
        type=str,
        help="Comma-separated boolean flags indicating if each objective is minimized (e.g., 'True,False')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots",
        help="Output directory for plots (default: ./plots)"
    )
    parser.add_argument(
        "--reference-point",
        type=str,
        default=None,
        help="Reference point for HV calculation as comma-separated values (e.g., '10,10'). If not provided, will be calculated automatically."
    )
    
    return parser.parse_args()


def parse_solution_file(file_path: Path) -> Dict[int, List[List[float]]]:
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
                    if current_iteration is not None:
                        try:
                            # Parse solution vector: "[1.3, 4]" -> [1.3, 4.0]
                            solution = ast.literal_eval(line)
                            if isinstance(solution, list):
                                solutions_by_iteration[current_iteration].append(solution)
                        except (ValueError, SyntaxError) as e:
                            print(f"Warning: Could not parse solution vector '{line}' in {file_path}: {e}")
                            continue
    
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
        return {}
    
    return solutions_by_iteration


def load_all_runs(algorithm_path: Path) -> Dict[str, Dict[int, List[List[float]]]]:
    """
    Load solutions from all run subfolders.
    
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
    
    print(f"Found {len(run_dirs)} run directories")
    
    for run_dir in run_dirs:
        solution_file = run_dir / "nn-solutions.txt"
        if solution_file.exists():
            print(f"Loading run: {run_dir.name}")
            solutions = parse_solution_file(solution_file)
            if solutions:
                all_runs[run_dir.name] = solutions
            else:
                print(f"Warning: No solutions found in {run_dir.name}")
        else:
            print(f"Warning: nn-solutions.txt not found in {run_dir}")
    
    return all_runs


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


def build_reference_pareto_set(all_runs: Dict[str, Dict[int, List[List[float]]]], 
                             minimization_flags: List[bool]) -> np.ndarray:
    """
    Build reference Pareto set from the final iteration solutions of all runs.
    """
    all_final_solutions = []
    
    for run_name, run_data in all_runs.items():
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
        raise ValueError("No final solutions found across all runs")
    
    # Find Pareto-optimal solutions
    all_final_array = np.array(all_final_solutions)
    pareto_mask = paretoset.paretoset(all_final_array, sense=["min"] * len(minimization_flags))
    reference_set = all_final_array[pareto_mask]
    
    print(f"Reference Pareto set contains {len(reference_set)} solutions")
    return reference_set


def calculate_reference_point(reference_set: np.ndarray, minimization_flags: List[bool]) -> np.ndarray:
    """
    Calculate reference point for hypervolume calculation.
    Uses maximum values with a small margin.
    """
    # For minimization problems, reference point should be worse (higher) than all solutions
    margins = np.abs(reference_set.max(axis=0) - reference_set.min(axis=0)) * 0.1
    margins = np.maximum(margins, 1.0)  # Ensure minimum margin
    reference_point = reference_set.max(axis=0) + margins
    
    print(f"Calculated reference point: {reference_point}")
    return reference_point


def calculate_metrics_over_time(all_runs: Dict[str, Dict[int, List[List[float]]]], 
                              reference_set: np.ndarray,
                              reference_point: np.ndarray,
                              minimization_flags: List[bool]) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Calculate HV and IGD metrics for each iteration across all runs.
    
    Returns:
        Tuple of (HV values by iteration, IGD values by iteration)
    """
    hv_by_iteration = defaultdict(list)
    igd_by_iteration = defaultdict(list)
    
    # Initialize metric calculators
    hv_indicator = HV(ref_point=reference_point)
    igd_indicator = IGD(reference_set)
    
    # Get all unique iterations across runs
    all_iterations = set()
    for run_data in all_runs.values():
        all_iterations.update(run_data.keys())
    all_iterations = sorted(all_iterations)
    
    print(f"Calculating metrics for {len(all_iterations)} iterations across {len(all_runs)} runs")
    
    for run_name, run_data in all_runs.items():
        print(f"Processing run: {run_name}")
        
        for iteration in all_iterations:
            if iteration in run_data and run_data[iteration]:
                # Convert solutions to minimization form
                solutions = convert_to_minimization(run_data[iteration], minimization_flags)
                
                # Calculate HV
                try:
                    hv_value = hv_indicator(solutions)
                    hv_by_iteration[iteration].append(hv_value)
                except Exception as e:
                    print(f"Warning: Could not calculate HV for run {run_name}, iteration {iteration}: {e}")
                
                # Calculate IGD
                try:
                    igd_value = igd_indicator(solutions)
                    igd_by_iteration[iteration].append(igd_value)
                except Exception as e:
                    print(f"Warning: Could not calculate IGD for run {run_name}, iteration {iteration}: {e}")
    
    return dict(hv_by_iteration), dict(igd_by_iteration)


def plot_convergence(hv_by_iteration: Dict[int, List[float]], 
                    igd_by_iteration: Dict[int, List[float]],
                    output_dir: Path,
                    algorithm_name: str):
    """
    Create convergence plots for HV and IGD metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    iterations = sorted(hv_by_iteration.keys())
    
    # Calculate statistics
    hv_means = [np.mean(hv_by_iteration[it]) for it in iterations]
    hv_stds = [np.std(hv_by_iteration[it]) for it in iterations]
    
    igd_means = [np.mean(igd_by_iteration[it]) for it in iterations]
    igd_stds = [np.std(igd_by_iteration[it]) for it in iterations]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    iterations = [it / iterations[0] for it in iterations]

    # Plot Hypervolume
    ax1.plot(iterations, hv_means, 'b-', linewidth=2, label='Mean HV')
    
    ax1.fill_between(iterations, 
                     np.array(hv_means) - np.array(hv_stds),
                     np.array(hv_means) + np.array(hv_stds),
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax1.set_xlabel('FE')
    ax1.set_ylabel('Hypervolume')
    ax1.set_title(f'SLB50 - Hypervolume Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Plot IGD
    ax2.plot(iterations, igd_means, 'r-', linewidth=2, label='Mean IGD')
    ax2.fill_between(iterations,
                     np.array(igd_means) - np.array(igd_stds),
                     np.array(igd_means) + np.array(igd_stds),
                     alpha=0.3, color='red', label='±1 Std Dev')
    ax2.set_xlabel('FE')
    ax2.set_ylabel('Inverted Generational Distance')
    ax2.set_title(f'SLB50 - IGD Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"{algorithm_name}_convergence.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {plot_path}")
    
    # Save data to CSV
    import pandas as pd
    df_data = {
        'Iteration': iterations,
        'HV_Mean': hv_means,
        'HV_Std': hv_stds,
        'IGD_Mean': igd_means,
        'IGD_Std': igd_stds
    }
    
    df = pd.DataFrame(df_data)
    csv_path = output_dir / f"{algorithm_name}_convergence_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Convergence data saved to: {csv_path}")
    
    plt.show()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Parse minimization flags
    try:
        minimization_flags = [flag.strip().lower() == 'true' 
                            for flag in args.minimization_flags.split(',')]
    except Exception as e:
        print(f"Error parsing minimization flags: {e}")
        print("Expected format: 'True,False' or 'true,false'")
        sys.exit(1)
    
    print(f"Minimization flags: {minimization_flags}")
    
    # Load algorithm path
    algorithm_path = Path(args.algorithm_path)
    algorithm_name = algorithm_path.name
    
    # Load all runs
    print("Loading experimental runs...")
    all_runs = load_all_runs(algorithm_path)
    
    if not all_runs:
        print("Error: No valid runs found")
        sys.exit(1)
    
    print(f"Successfully loaded {len(all_runs)} runs")
    
    # Build reference Pareto set
    print("Building reference Pareto set...")
    try:
        reference_set = build_reference_pareto_set(all_runs, minimization_flags)
    except Exception as e:
        print(f"Error building reference set: {e}")
        sys.exit(1)
    
    # Calculate or parse reference point
    if args.reference_point:
        try:
            reference_point = np.array([float(x.strip()) for x in args.reference_point.split(',')])
            print(f"Using provided reference point: {reference_point}")
        except Exception as e:
            print(f"Error parsing reference point: {e}")
            print("Calculating reference point automatically...")
            reference_point = calculate_reference_point(reference_set, minimization_flags)
    else:
        reference_point = calculate_reference_point(reference_set, minimization_flags)
    
    # Calculate metrics over time
    print("Calculating metrics over time...")
    hv_by_iteration, igd_by_iteration = calculate_metrics_over_time(
        all_runs, reference_set, reference_point, minimization_flags
    )
    
    if not hv_by_iteration or not igd_by_iteration:
        print("Error: No metrics calculated")
        sys.exit(1)
    
    # Create plots
    print("Creating convergence plots...")
    output_dir = Path(args.output_dir)
    plot_convergence(hv_by_iteration, igd_by_iteration, output_dir, algorithm_name)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
