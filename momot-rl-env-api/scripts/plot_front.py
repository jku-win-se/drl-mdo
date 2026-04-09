import sys
import matplotlib.pyplot as plt
from paretoset import paretoset
import numpy as np
import os
from matplotlib import rcParams
rcParams['axes.titlepad'] = 60

tick_size = 30
label_size = 30
m_size = 390

labels = {
    "NSGAII": "NSGA-II",

    "PQ-HV": "HV-PQL",
    "PQ-PO": "PO-PQL",
    "Chebycheff": "Chebyshev-QL",
    "Tournament": "NW-QL",
    "PQ-HV-Local10": "HV-PQL",
    "PQ-PO-Local10": "PO-PQL",
    "Chebycheff-10": "Chebyshev-QL",
    "Tournament-Local10": "NW-QL",
    "QLearning": "Q",
    "QLearningExplore": "Q$^+$",
    "GNN-PPO": "GNN-PPO"
}
# Dictionary to hold solutions for each algorithm
algorithm_solutions = {}

# Function to parse objective vectors from lines


def build_pareto_set(solutions, dims):
    """
    Build the Pareto set from the given list of solutions.
    """
    solutions = solutions[:, [*dims]]

    mask = paretoset(solutions, sense=["min", "min"])
    return solutions[mask]


def parse_vector(line):
    return tuple(map(float, [val.strip('[],') for val in line.split()]))


if len(sys.argv) < 5:
    print("Usage: python script.py <filename> <dimension1> <dimension2> <xlabel> <ylabel> <title> <greedy_algs> <algorithm1> <algorithm2> ...")
    sys.exit(1)

filename = sys.argv[1]
dimension1 = int(sys.argv[2])
dimension2 = int(sys.argv[3])
xlabel = sys.argv[4]
ylabel = sys.argv[5]
title = sys.argv[6]

algorithms = sys.argv[8:]
is_greedy = int(sys.argv[7])

# Determine the directory of the provided file
file_dir = os.path.dirname(os.path.abspath(filename))
output_file = os.path.join(file_dir, '..',  f"{title}.pdf")

if is_greedy:
    for k in labels:
        if k != 'NSGAII-2':
            labels[k] += '$^+$'
# Open the file
print(algorithms)
with open(filename, "r") as file:
    current_algorithm = None
    current_solutions = []
    in_solutions_section = False

    # Iterate through each line in the file

    for line in file:
        line = line.strip()
        print(line)
        # Check if line contains algorithm name
        if line in algorithms:
            print("got")
            if in_solutions_section:
                algorithm_solutions[current_algorithm] = current_solutions
                in_solutions_section = False
            current_algorithm = line.strip()
            current_solutions = []
            in_solutions_section = True
        elif in_solutions_section and line:
            # Parse objective vector and add to current solutions
            current_solutions.append(parse_vector(line))
        else:
            # Empty line indicates end of solutions for current algorithm
            if in_solutions_section:
                algorithm_solutions[current_algorithm] = current_solutions
                in_solutions_section = False
            current_algorithm = None
            current_solutions = []

# Plotting
plt.figure(figsize=(8, 6))

# Define markers and colors for each algorithm
markers = ['o', 's', 'X', 'p', '^']
m_sizes = [m_size, m_size, m_size-150, m_size, m_size]
facecolors = ['antiquewhite', 'peachpuff', 'thistle', 'lightgreen', 'peru', 'crimson', 'navy']
print(algorithms)
for i, algorithm in enumerate(algorithms):

    solutions = algorithm_solutions.get(algorithm, [])
    print(algorithm, solutions)
    if len(solutions[0]) > 2:
        solutions = build_pareto_set(
            np.array(solutions), sorted([dimension1, dimension2]))
   # print(solutions[:3])
    xs = [solution[dimension1-1] for solution in solutions]
   # print(xs[:3])
    ys = [solution[dimension2-1] for solution in solutions]
    if algorithm == 'NSGAII-2':
        plt.scatter(xs, ys, color='blue', marker='o',
                    label=labels[algorithm],  facecolor='lightblue', edgecolor='blue', s=m_size, alpha=.65)
    else:
        print(i)
        plt.scatter(xs, ys, edgecolor='darkorange', facecolor=facecolors[i % len(markers)],
                    marker=markers[i % len(markers)], label=labels[algorithm] if algorithm in labels else algorithm, s=m_sizes[i % len(markers)], alpha=.65)

plt.locator_params(axis='x', nbins=5)

plt.title(title)
plt.xlabel(xlabel, fontsize=label_size)
plt.ylabel(ylabel, fontsize=label_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
leg = plt.legend(loc='upper right', fontsize=label_size, )
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor((1, 1, 1, 0.7))
plt.tight_layout()

# Save the plot as a high-resolution PDF file
plt.savefig(output_file, dpi=300)

plt.show()