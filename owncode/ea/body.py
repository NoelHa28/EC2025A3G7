from opposites import has_core_opposite_pair, simple_symmetry_score
from evaluate import STOCHASTIC_SPAWN_POSITIONS, set_spawn_position

KILL_FITNESS = -100.0  # you already filter this out in selection


from typing import Any
from collections.abc import Callable
import multiprocessing as mp

import numpy as np
from robot import Robot

SEED = 42
RNG = np.random.default_rng(SEED)

from .mind import MindEA

type Genotype = list[np.ndarray]


class Mutate:
    def __init__(self, mutation_rate: float = 0.1) -> None:
        self.mutation_rate = mutation_rate

    def gaussian(self, genotype: Genotype) -> Genotype:
        mutated_genotype = []

        for gene in genotype:
            # Create a copy to avoid modifying the original
            mutated = gene.copy()
            # Create a mask for which genes to mutate
            mutation_mask = RNG.random(len(gene)) < self.mutation_rate

            # Apply Gaussian noise to selected genes
            if np.any(mutation_mask):
                noise = RNG.normal(0, 0.05, size=len(gene)).astype(np.float32)
                mutated[mutation_mask] += noise[mutation_mask]

                # Clip values to stay within [0, 1] bounds
                mutated = np.clip(mutated, 0.0, 1.0)

            mutated_genotype.append(mutated)

        return mutated_genotype


class Crossover:
    def __init__(self, crossover_type: str = "onepoint") -> None:
        self.crossover_type = crossover_type

    def __call__(
        self, parent1: Genotype, parent2: Genotype
    ) -> tuple[Genotype, Genotype]:
        if self.crossover_type == "onepoint":
            return self._onepoint_crossover(parent1, parent2)
        elif self.crossover_type == "uniform":
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_type == "blend":
            return self._blend_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}")

    def _onepoint_crossover(
        self, parent1: Genotype, parent2: Genotype
    ) -> tuple[Genotype, Genotype]:
        offspring1, offspring2 = [], []

        for gene1, gene2 in zip(parent1, parent2):
            # Choose a random crossover point
            crossover_point = RNG.integers(1, len(gene1))

            # Create offspring by swapping after crossover point
            child1 = np.concatenate([gene1[:crossover_point], gene2[crossover_point:]])
            child2 = np.concatenate([gene2[:crossover_point], gene1[crossover_point:]])

            offspring1.append(child1)
            offspring2.append(child2)

        return offspring1, offspring2

    def _uniform_crossover(
        self, parent1: Genotype, parent2: Genotype
    ) -> tuple[Genotype, Genotype]:
        offspring1, offspring2 = [], []

        for gene1, gene2 in zip(parent1, parent2):
            # Create random mask for gene selection
            mask = RNG.random(len(gene1)) < 0.5

            # Create offspring using the mask
            child1 = np.where(mask, gene1, gene2)
            child2 = np.where(mask, gene2, gene1)

            offspring1.append(child1)
            offspring2.append(child2)

        return offspring1, offspring2

    def _blend_crossover(
        self, parent1: Genotype, parent2: Genotype, alpha: float = 0.5
    ) -> tuple[Genotype, Genotype]:
        offspring1, offspring2 = [], []

        for gene1, gene2 in zip(parent1, parent2):
            # Calculate min and max values for each gene position
            min_vals = np.minimum(gene1, gene2)
            max_vals = np.maximum(gene1, gene2)

            # Calculate range and extend it by alpha
            gene_range = max_vals - min_vals
            extended_min = min_vals - alpha * gene_range
            extended_max = max_vals + alpha * gene_range

            # Generate random values within the extended range
            child1 = RNG.uniform(extended_min, extended_max).astype(np.float32)
            child2 = RNG.uniform(extended_min, extended_max).astype(np.float32)

            # Clip to maintain [0, 1] bounds
            child1 = np.clip(child1, 0.0, 1.0)
            child2 = np.clip(child2, 0.0, 1.0)

            offspring1.append(child1)
            offspring2.append(child2)

        return offspring1, offspring2


class BodyEA:
    def __init__(
        self,
        population_size: int,
        generations: int,
        genotype_size: int,
        evaluator: Callable[[Any], float],
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        crossover_type: str = "onepoint",
        elitism: int = 1,
        selection: str = "tournament",
        tournament_size: int = 3,
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
        self.population_size = population_size
        self.generations = generations
        self.genotype_size = genotype_size
        self.evaluator = evaluator  # Function to evaluate fitness
        self.mutate = Mutate(mutation_rate)
        self.crossover = Crossover(crossover_type)
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.selection = selection
        self.tournament_size = tournament_size

    def random_genotype(self) -> Genotype:
        """Generate a random genotype."""
        type_p_genes = RNG.random(self.genotype_size).astype(np.float32)
        conn_p_genes = RNG.random(self.genotype_size).astype(np.float32)
        rot_p_genes = RNG.random(self.genotype_size).astype(np.float32)

        return [type_p_genes, conn_p_genes, rot_p_genes]

    def select_parents(
        self, population: list[Genotype], fitness_scores: list[float]
    ) -> tuple[Genotype, Genotype]:
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
        viable_indices = [
            i for i, fitness in enumerate(fitness_scores) if fitness != -100.0
        ]

        # Safety check: ensure we have at least 2 viable individuals
        if len(viable_indices) < 2:
            # Fallback: use all individuals if too few viable ones
            print(
                f"  → Warning: Only {len(viable_indices)} viable individuals, using all for selection"
            )
            viable_indices = list(range(len(population)))

        viable_population = [population[i] for i in viable_indices]
        viable_fitness_scores = [fitness_scores[i] for i in viable_indices]

        if self.selection == "tournament":
            parent1 = self._tournament_selection(
                viable_population, viable_fitness_scores
            )
            parent2 = self._tournament_selection(
                viable_population, viable_fitness_scores
            )
            return parent1, parent2
        elif self.selection == "roulette":
            parent1 = self._roulette_selection(viable_population, viable_fitness_scores)
            parent2 = self._roulette_selection(viable_population, viable_fitness_scores)
            return parent1, parent2
        else:
            raise ValueError(f"Unknown selection method: {self.selection}")

    def _tournament_selection(
        self, population: list[Genotype], fitness_scores: list[float]
    ) -> Genotype:
        """
        Tournament selection: Pick the best individual from a random tournament.

        Args:
            population: List of genotypes
            fitness_scores: List of fitness scores (higher = better)

        Returns:
            Selected genotype
        """
        # Select random tournament individuals
        tournament_indices = RNG.choice(
            len(population), size=self.tournament_size, replace=False
        )

        # Find the best individual in the tournament (highest fitness)
        best_index = tournament_indices[0]
        best_fitness = fitness_scores[best_index]

        for idx in tournament_indices[1:]:
            if fitness_scores[idx] > best_fitness:
                best_fitness = fitness_scores[idx]
                best_index = idx

        return population[best_index]

    def _roulette_selection(
        self, population: list[Genotype], fitness_scores: list[float]
    ) -> Genotype:
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
        fitness_scores = []
        for genotype in population:
            fitness = self._eval_func(genotype)
            fitness_scores.append(fitness)
        return fitness_scores

    def _eval_func(self, genotype: Genotype) -> float:
        # Build morphology
        try:
            robot = Robot(genotype)
        except RuntimeError:
            return KILL_FITNESS  # e.g., no hinges

        G = robot.graph

        # Cheap hard gate (optional, keeps cost low)
        if not has_core_opposite_pair(G):
            print("KILL (no opposite pair on core)")
            return KILL_FITNESS

        # Simple whole-body symmetry score (0..1)
        score = simple_symmetry_score(G, max_depth=3)

        # Probabilistic kill: worse symmetry -> higher chance to cull
        threshold = 0.6  # raise to be stricter (e.g., 0.7)
        softness = 0.3  # raise to soften ramp (e.g., 0.4)
        p_kill = max(0.0, min(1.0, (threshold - score) / max(softness, 1e-6)))

        # Quick debug line
        print(f"sym={score:.3f}  p_kill={p_kill:.2f}")

        if RNG.random() < p_kill:
            print("KILL (probabilistic)")
            return KILL_FITNESS

        # Survives -> evaluate brain as before
        ea = MindEA(
            robot=robot,
            population_size=20,
            generations=1,
            mutation_rate=0.5,
            crossover_rate=0.0,
            crossover_type="onepoint",
            elitism=50,
            selection="tournament",
            tournament_size=5,
        )
        _, weights, _ = ea.run()
        return float(max(weights))

        # def _eval_func(self, genotype: Genotype) -> float:
        ea = MindEA(
            robot=Robot(genotype),
            population_size=20,
            generations=1,
            mutation_rate=0.5,
            crossover_rate=0.0,
            crossover_type="onepoint",
            elitism=50,  # Keep top 10%
            selection="tournament",
            tournament_size=5,
        )
        _, weights, _ = ea.run()

        return max(weights)

    def create_initial_population(self) -> list[Genotype]:
        return [self.random_genotype() for _ in range(self.population_size)]

    def apply_elitism(
        self, population: list[Genotype], fitness_scores: list[float]
    ) -> list[Genotype]:
        if self.elitism <= 0:
            return []

        # Get indices sorted by fitness (highest first)
        elite_indices = np.argsort(fitness_scores)[-self.elitism :]
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
            print(f"Generation {generation+1}/{self.generations}")
            current_spawn = STOCHASTIC_SPAWN_POSITIONS[
                generation % len(STOCHASTIC_SPAWN_POSITIONS)
            ]
            set_spawn_position(current_spawn)
            print(f"[GEN {generation+1}] using spawn {current_spawn}", flush=True)
            # Evaluate population
            fitness_scores = self.evaluate_population(population)

            # Track statistics
            best_fitness = max(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness_history.append(best_fitness)
            average_fitness_history.append(average_fitness)

            # Print progress
            if generation % 10 == 0 or generation == self.generations - 1:
                print(
                    f"Generation {generation}: Best={best_fitness:.4f}, Avg={average_fitness:.4f}"
                )

            # Apply elitism (preserve best individuals)
            elite_individuals = self.apply_elitism(population, fitness_scores)

            # Create new population
            new_population = elite_individuals.copy()

            # Fill rest of population with offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitness_scores)

                # Apply crossover
                if RNG.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    # If no crossover, children are copies of parents
                    child1, child2 = [gene.copy() for gene in parent1], [
                        gene.copy() for gene in parent2
                    ]

                # Apply mutation
                child1 = self.mutate.gaussian(child1)
                child2 = self.mutate.gaussian(child2)

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
