#!/usr/bin/env python3
"""
Pareto Set Computation Script

Computes Pareto sets of solution vectors from text files across multiple algorithms.
Each algorithm path contains subfolders with "nn-solutions_final.txt" files.
"""

import os
import sys
import ast
import argparse
from pathlib import Path
from typing import List, Tuple, Set
import json


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Pareto sets from solution vectors in text files"
    )
    parser.add_argument(
        "iteration", 
        type=str, 
        help="Target iteration number to extract solutions from"
    )
    parser.add_argument(
        "algorithm_paths", 
        nargs="+", 
        help="Paths to algorithm directories (variable length)"
    )
    parser.add_argument(
        "--objectives", "-obj",
        type=str,
        required=True,
        help="Optimization direction for each dimension (e.g., 'min,max,min' for 3D problem)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="pareto_results.json",
        help="Output file for Pareto sets (default: pareto_results.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def parse_solution_file(file_path: str, target_iteration: int) -> List[List[float]]:
    """
    Parse a solutions file and extract solution vectors for the target iteration.
    
    Args:
        file_path: Path to the solutions file
        target_iteration: The iteration number to extract solutions for
        
    Returns:
        List of solution vectors (each vector is a list of floats)
    """
    solutions = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
                
            # Check if line contains the target iteration number
            try:
                if int(line) == target_iteration:
                    i += 1  # Move to next line
                    
                    # Parse solution vectors until empty line or end of file
                    while i < len(lines):
                        line = lines[i].strip()
                        
                        # Stop at empty line
                        if not line:
                            break
                            
                        # Parse solution vectors from this line using regex
                        import re
                        vector_pattern = r'\[([^\]]+)\]'
                        vector_matches = re.findall(vector_pattern, line)
                        
                        for vector_str in vector_matches:
                            try:
                                # Parse the numbers inside the brackets
                                numbers = [float(x.strip()) for x in vector_str.split(',')]
                                solutions.append(numbers)
                            except ValueError:
                                # Skip malformed vectors
                                pass
                                
                        i += 1
                    break
                else:
                    i += 1
            except ValueError:
                # Line doesn't start with an integer
                i += 1
                
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        
    return solutions


def parse_objectives(objectives_str: str) -> List[str]:
    """
    Parse the objectives string into a list of optimization directions.
    
    Args:
        objectives_str: Comma-separated string of 'min' or 'max' (e.g., 'min,max,min')
        
    Returns:
        List of optimization directions ['min', 'max', 'min']
    """
    objectives = [obj.strip().lower() for obj in objectives_str.split(',')]
    
    # Validate objectives
    for i, obj in enumerate(objectives):
        if obj not in ['min', 'max', 'minimize', 'maximize']:
            if obj.startswith('min'):
                objectives[i] = 'min'
            elif obj.startswith('max'):
                objectives[i] = 'max'
            else:
                raise ValueError(f"Invalid objective '{obj}'. Use 'min' or 'max'.")
        elif obj in ['minimize']:
            objectives[i] = 'min'
        elif obj in ['maximize']:
            objectives[i] = 'max'
    
    return objectives


def dominates(sol1: List[float], sol2: List[float], objectives: List[str]) -> bool:
    """
    Check if sol1 dominates sol2 with per-dimension optimization directions.
    
    Args:
        sol1: First solution vector
        sol2: Second solution vector
        objectives: List of optimization directions ('min' or 'max') for each dimension
        
    Returns:
        True if sol1 dominates sol2
    """
    if len(sol1) != len(sol2) or len(sol1) != len(objectives):
        return False
        
    at_least_one_better = False
    
    for v1, v2, obj in zip(sol1, sol2, objectives):
        if obj == 'min':
            # For minimization: sol1 must be <= sol2
            if v1 > v2:
                return False
            if v1 < v2:
                at_least_one_better = True
        else:  # obj == 'max'
            # For maximization: sol1 must be >= sol2  
            if v1 < v2:
                return False
            if v1 > v2:
                at_least_one_better = True
                
    return at_least_one_better


def compute_pareto_set(solutions: List[List[float]], objectives: List[str]) -> List[List[float]]:
    """
    Compute the Pareto set from a list of solution vectors.
    
    Args:
        solutions: List of solution vectors
        objectives: List of optimization directions ('min' or 'max') for each dimension
        
    Returns:
        List of non-dominated solutions (Pareto set)
    """
    if not solutions:
        return []
        
    # Validate that all solutions have the same dimensionality as objectives
    for sol in solutions:
        if len(sol) != len(objectives):
            raise ValueError(f"Solution dimension {len(sol)} doesn't match objectives dimension {len(objectives)}")
        
    pareto_set = []
    
    for candidate in solutions:
        is_dominated = False
        
        # Check if candidate is dominated by any solution in current Pareto set
        for pareto_sol in pareto_set:
            if dominates(pareto_sol, candidate, objectives):
                is_dominated = True
                break
                
        if not is_dominated:
            # Remove solutions from Pareto set that are dominated by candidate
            pareto_set = [sol for sol in pareto_set if not dominates(candidate, sol, objectives)]
            pareto_set.append(candidate)
            
    return pareto_set


def process_algorithm(algorithm_path: str, target_iteration: int, objectives: List[str], verbose: bool = False) -> List[List[float]]:
    """
    Process all subfolders of an algorithm path to build aggregate Pareto set.
    
    Args:
        algorithm_path: Path to algorithm directory
        target_iteration: Target iteration number
        objectives: List of optimization directions for each dimension
        verbose: Enable verbose output
        
    Returns:
        Aggregate Pareto set for the algorithm
    """
    algorithm_path = Path(algorithm_path)
    all_solutions = []
    processed_files = 0
    
    if verbose:
        print(f"Processing algorithm: {algorithm_path}")
        print(f"  Objectives: {objectives}")
    
    # Traverse all subfolders
    for subfolder in algorithm_path.rglob("*"):
        if subfolder.is_dir():
            solution_file = subfolder / "nn-solutions.txt"
            
            if solution_file.exists():
                if verbose:
                    print(f"  Processing: {solution_file}")
                    
                solutions = parse_solution_file(str(solution_file), target_iteration)
                
                # Validate solution dimensions
                valid_solutions = []
                for sol in solutions:
                    if len(sol) == len(objectives):
                        valid_solutions.append(sol)
                    elif verbose:
                        print(f"    Warning: Skipping solution with {len(sol)} dimensions (expected {len(objectives)})")
                
                all_solutions.extend(valid_solutions)
                processed_files += 1
                
                if verbose:
                    print(f"    Found {len(valid_solutions)} valid solutions for iteration {target_iteration}")
    
    if verbose:
        print(f"  Processed {processed_files} files, total {len(all_solutions)} solutions")
    
    # Compute Pareto set from all collected solutions
    pareto_set = compute_pareto_set(all_solutions, objectives)
    
    if verbose:
        print(f"  Pareto set size: {len(pareto_set)}")
        
    return pareto_set


def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse objectives
    try:
        objectives = parse_objectives(args.objectives)
    except ValueError as e:
        print(f"Error parsing objectives: {e}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Target iteration: {args.iteration}")
        print(f"Objectives: {objectives}")
        print(f"Algorithm paths: {args.algorithm_paths}")
        print(f"Output file: {args.output}")
        print()
    
    results = {}
    iters = [int(obj.strip().lower()) for obj in args.iteration.split(',')]


    # Process each algorithm
    for (iter, algo_path) in zip(iters, args.algorithm_paths):
        algo_name = Path(algo_path).name
        
        try:
            pareto_set = process_algorithm(algo_path, iter, objectives, args.verbose)
            print(pareto_set)
            # Write the elements of pareto_set line by line to text file in subfolder named to algo_name
       
            results[f"{algo_name}_{iter}"] = {
                'path': algo_path,
                'iteration': iter,
                'pareto_set': pareto_set,
                'pareto_size': len(pareto_set)
            }
        except Exception as e:
            print(f"Error processing algorithm '{algo_name}': {e}")
            results[f"{algo_name}_{iter}"] = {
                'path': algo_path,
                'error': str(e),
                'iteration': iter,
                'pareto_set': [],
                'pareto_size': 0
            }


    output_file = f"pareto_set.txt"
    with open(output_file, 'w') as f:
        for r in results.values():
            f.write(f"{r['iteration']}\n")
            for sol in r['pareto_set']:
                f.write(f"{sol}\n")
            f.write("\n")

if __name__ == "__main__":
    main()