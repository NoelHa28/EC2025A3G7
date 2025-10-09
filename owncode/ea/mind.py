from typing import Any
from collections.abc import Callable
import multiprocessing as mp

import numpy as np
import controller
from robot import Robot
from evaluate import evaluate
import math

CTX = mp.get_context("spawn")
CPU_COUNT = mp.cpu_count()

from consts import RNG, DATA

type Genotype = np.ndarray

def solve_for_n(x):
    """
    Solves 3N^2 + 2N = X for the positive N.
    Returns a single positive float.
    """
    discriminant = 4 + 12 * x
    if discriminant < 0:
        return None  # no real solutions
    
    sqrt_disc = math.sqrt(discriminant)
    n = (-2 + sqrt_disc) / 6
    return n if n > 0 else None

class Mutation:
    def __init__(self, mutation_rate: float, n: int):
        self.mutation_rate = mutation_rate
        self.n = n

    def __call__(self, genotype: Genotype) -> Genotype:
        # genotype[:self.n] = self.bit_flip(genotype[:self.n])
        genotype[self.n:] = self.gaussian(genotype[self.n:])

        return genotype.clip(0.0, 1.0)

    def bit_flip(self, genotype: Genotype) -> Genotype:
        """Apply bit-flip mutation to the genotype."""
        for i in range(len(genotype)):
            if RNG.random() < self.mutation_rate:
                genotype[i] = 1.0 - genotype[i]  # Flip between 0 and 1
        return genotype

    def gaussian(self, genotype: Genotype) -> Genotype:
        """Apply Gaussian mutation to the genotype."""
        if RNG.random() < self.mutation_rate:
            # Apply random mutation
            mutation = RNG.normal(0, 0.05, size=genotype.shape).astype(np.float32)
            genotype += mutation
        return genotype

class Crossover:
    def __init__(self, crossover_type: str, n: int):
        self.crossover_type = crossover_type
        self.n = n

    def __call__(self, parent1: Genotype, parent2: Genotype) -> tuple[Genotype, Genotype]:
        if self.crossover_type == "onepoint":
            return self.one_point(parent1, parent2)
        elif self.crossover_type == "uniform":
            return self.uniform(parent1, parent2)
        elif self.crossover_type == "blend":
            return self.blend(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}")

    def one_point(self, parent1: Genotype, parent2: Genotype) -> tuple[Genotype, Genotype]:
        """One-point crossover between two parents."""
        point = RNG.integers(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1.astype(np.float32), child2.astype(np.float32)

    def uniform(self, parent1: Genotype, parent2: Genotype) -> tuple[Genotype, Genotype]:
        """Uniform crossover between two parents."""
        mask = RNG.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1.astype(np.float32), child2.astype(np.float32)

    def blend(self, parent1: Genotype, parent2: Genotype, alpha: float = 0.5) -> tuple[Genotype, Genotype]:
        """Blend crossover between two parents."""
        child1 = (alpha * parent1 + (1 - alpha) * parent2).astype(np.float32)
        child2 = (alpha * parent2 + (1 - alpha) * parent1).astype(np.float32)

        child1[self.n:] = child1[self.n:].round()
        child2[self.n:] = child2[self.n:].round()
        return child1, child2

class MindEA:
    def __init__(
        self,
        robot: Robot,
        population_size: int,
        generations: int,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        crossover_type: str = "onepoint",
        elitism: int = 1,
        selection: str = "tournament",
        tournament_size: int = 3
    ) -> None:
        """
        Simple evolutionary algorithm implementation.
        
        Args:
            population_size (int): Number of individuals in the population.
            generations (int): Number of generations to run the algorithm.
            genotype_size (int): Size of the genotype (number of genes).
            evaluator (callable): Function to evaluate the fitness of an individual.
            mutation_rate (float): Probability of mutation for each gene.
            crossover_rate (float): Probability of crossover between two parents.
            crossover_type (str): Type of crossover ("onepoint", "uniform", or "blend").
            elitism (int): Number of top individuals to carry over to the next generation.
            selection (str): Selection method ("tournament" or "roulette").
            tournament_size (int): Size of the tournament for tournament selection.
        """
        self.robot = robot

        connection_params = robot._number_of_hinges ** 2

        self.population_size = population_size
        self.generations = generations
        self.mutate = Mutation(mutation_rate, connection_params)
        self.crossover = Crossover(crossover_type, connection_params)
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.selection = selection
        self.tournament_size = tournament_size
    
    def random_genotype(self) -> Genotype:
        return self.robot.brain.generate_random_cpg_genotype(self.robot._number_of_hinges)
    
    def select_parents(self, population: list[Genotype], fitness_scores: list[float]) -> tuple[Genotype, Genotype]:
        """
        Select two parents from the population based on the specified selection method.
        Excludes individuals with fitness = -100 (failed simulations) from selection.
        
        Args:
            population: List of genotypes in the current population
            fitness_scores: List of fitness scores corresponding to each genotype
            
        Returns:
            tuple: Two selected parent genotypes
        """
        # Filter out failed individuals (fitness = -100) for parent selection
        viable_indices = [i for i, fitness in enumerate(fitness_scores) if fitness != -100.0]
        
        # Safety check: ensure we have at least 2 viable individuals
        if len(viable_indices) < 2:
            # Fallback: use all individuals if too few viable ones
            print(f"  → Warning: Only {len(viable_indices)} viable individuals, using all for selection")
            viable_indices = list(range(len(population)))
        
        viable_population = [population[i] for i in viable_indices]
        viable_fitness_scores = [fitness_scores[i] for i in viable_indices]
        
        if self.selection == "tournament":
            parent1 = self._tournament_selection(viable_population, viable_fitness_scores)
            parent2 = self._tournament_selection(viable_population, viable_fitness_scores)
            return parent1, parent2
        elif self.selection == "roulette":
            parent1 = self._roulette_selection(viable_population, viable_fitness_scores)
            parent2 = self._roulette_selection(viable_population, viable_fitness_scores)
            return parent1, parent2
        else:
            raise ValueError(f"Unknown selection method: {self.selection}")

    def _tournament_selection(self, population: list[Genotype], fitness_scores: list[float]) -> Genotype:
        """
        Tournament selection: Pick the best individual from a random tournament.
        
        Args:
            population: List of genotypes
            fitness_scores: List of fitness scores (higher = better)
            
        Returns:
            Selected genotype
        """
        # Select random tournament individuals
        tournament_indices = RNG.choice(len(population), size=self.tournament_size, replace=False)
        
        # Find the best individual in the tournament (highest fitness)
        best_index = tournament_indices[0]
        best_fitness = fitness_scores[best_index]
        
        for idx in tournament_indices[1:]:
            if fitness_scores[idx] > best_fitness:
                best_fitness = fitness_scores[idx]
                best_index = idx
        
        return population[best_index]
    
    def _roulette_selection(self, population: list[Genotype], fitness_scores: list[float]) -> Genotype:
        """
        Roulette wheel selection: Select individuals proportional to their fitness.
        Uses rank-based selection to handle negative fitness values while preserving original fitness.
        
        Args:
            population: List of genotypes
            fitness_scores: List of fitness scores
            
        Returns:
            Selected genotype
        """
        # Sort indices by fitness (lowest to highest)
        sorted_indices = np.argsort(fitness_scores)
        
        # Assign ranks (1 = worst, population_size = best)
        ranks = np.zeros(len(fitness_scores))
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = i + 1  # Rank starts from 1
        
        # Calculate selection probabilities based on ranks
        total_rank = sum(ranks)
        probabilities = ranks / total_rank
        
        # Select individual based on probabilities
        selected_index = RNG.choice(len(population), p=probabilities)
        return population[selected_index]
    
    def evaluate_population(self, population: list[Genotype]) -> list[float]:
        with CTX.Pool(CPU_COUNT) as pool:
            fitness_scores = pool.map(self._eval_func, population)
        return fitness_scores

    def _eval_func(self, genotype: Genotype) -> Any:
        self.robot.brain.set_genotype(genotype)
        return evaluate(self.robot)

    def create_initial_population(self) -> list[Genotype]:
        return [self.random_genotype() for _ in range(self.population_size)]

    def apply_elitism(self, population: list[Genotype], fitness_scores: list[float]) -> list[Genotype]:
        if self.elitism <= 0:
            return []
        
        # Get indices sorted by fitness (highest first)
        elite_indices = np.argsort(fitness_scores)[-self.elitism:]
        elite_individuals = [population[i] for i in elite_indices]
        
        return elite_individuals
    
    def run(self) -> tuple[Genotype, list[float], list[float]]:
        """
        Run the evolutionary algorithm.
        
        Returns:
            tuple: (best_individual, best_fitness_history, average_fitness_history)
        """
        # Initialize population
        population = self.create_initial_population()
        
        # Track statistics
        best_fitness_history = []
        average_fitness_history = []
        
        for generation in range(self.generations):
            # Evaluate population
            fitness_scores = self.evaluate_population(population)
            
            # Track statistics
            best_fitness = max(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness_history.append(best_fitness)
            average_fitness_history.append(average_fitness)
            
            # Print progress
            print(f"Generation {generation + 1}: Best={best_fitness:.4f}, Avg={average_fitness:.4f}")
            
            # Apply elitism (preserve best individuals)
            elite_individuals = self.apply_elitism(population, fitness_scores)
            
            # Create new population
            new_population = elite_individuals.copy()
            
            # Fill rest of population with offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = parent1.copy(), parent2.copy()
                
                # Apply crossover
                if RNG.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)

                # Apply mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add to new population (check size to avoid exceeding population_size)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Update population
            population = new_population
        
        # Return best individual from final generation
        final_fitness_scores = self.evaluate_population(population)
        best_index = np.argmax(final_fitness_scores)
        best_individual = population[best_index]
        
        return best_individual, best_fitness_history, average_fitness_history
                