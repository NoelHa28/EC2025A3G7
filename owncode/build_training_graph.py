import os
import pickle
import re

from robot import Robot
from matplotlib import pyplot as plt
import numpy as np

def read_generations(directory):
    """
    Reads all files ending with _gen{gen_num}.pkl in the given directory and builds a dict:
    {gen_num: (body_population, body_fitness)}
    """
    gen_dict = {}
    pop_pattern = re.compile(r'body_population_gen(\d+)\.pkl$')
    fit_pattern = re.compile(r'body_fitness_scores_gen(\d+)\.pkl$')

    # Find all relevant files
    files = os.listdir(directory)
    pop_files = {int(pop_pattern.match(f).group(1)): f for f in files if pop_pattern.match(f)}
    fit_files = {int(fit_pattern.match(f).group(1)): f for f in files if fit_pattern.match(f)}

    # Only include generations where both files exist
    common_gens = set(pop_files.keys()) & set(fit_files.keys())

    for gen_num in sorted(common_gens):
        with open(os.path.join(directory, pop_files[gen_num]), 'rb') as f_pop:
            body_population = [Robot(body_genotype=genotype) for genotype in pickle.load(f_pop)]
        with open(os.path.join(directory, fit_files[gen_num]), 'rb') as f_fit:
            body_fitness = pickle.load(f_fit)
        gen_dict[gen_num] = (body_population, body_fitness)

    return gen_dict

data = read_generations('run2')


generations = sorted(data.keys())
fitness_lists = []
for g in generations:
    fitness = []

    for f in data[g][1]:
        if f > -7:  # Filter out invalid fitness scores
            fitness.append(f)

    fitness_lists.append(fitness)

means = [np.mean(f) for f in fitness_lists]
stds = [np.std(f) for f in fitness_lists]

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(generations, means, marker='o', markersize=2, label='Average Fitness', color='#f177c2')
plt.fill_between(generations, 
                 np.array(means) - np.array(stds),
                 np.array(means) + np.array(stds),
                 alpha=.15, color='#f177c2', label='±1 SD')

plt.title("Evolution of Fitness Across Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('evolution_fitness_plot.png', dpi=300)
plt.show()