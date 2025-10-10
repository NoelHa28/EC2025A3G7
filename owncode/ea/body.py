import pickle as pkl
from typing import Any
from collections.abc import Callable

from multiprocessing import Pool
import numpy as np

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import draw_graph, save_graph_as_json

import controller
from evaluate import evaluate
from morphology_constraints import is_robot_viable
from consts import STOCHASTIC_SPAWN_POSITIONS, RNG, GENOTYPE_SIZE
from .mind import MindEA
from robot import Robot

type Genotype = list[np.ndarray]

def bouncy_clip(genotype: np.ndarray, limit: int) -> Genotype:
    genotype = ((genotype + limit) % (2 * limit)) - limit
    genotype[genotype > limit] = 2 * limit - genotype[genotype > limit]
    return genotype

def calculate_slope_of_weights(weights: list[float]) -> float:
    if len(weights) < 2:
        return 0.0

    start_weights = weights[0]
    end_weights = weights[-1]
    slope = (end_weights - start_weights) / (len(weights) - 1)
    return slope

def random_genotype() -> Genotype:
    return [
        RNG.uniform(-1, 1, GENOTYPE_SIZE).astype(np.float32),
        RNG.uniform(-1, 1, GENOTYPE_SIZE).astype(np.float32),
        RNG.uniform(-1, 1, GENOTYPE_SIZE).astype(np.float32),
    ]

def _evaluate_genotype(_):
    genotype = random_genotype()
    success = evaluate(
        Robot(body_genotype=genotype),
        learn_test=True,
        controller_func=controller.random
    )
    return genotype if success else None

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

                noise = RNG.normal(0, 0.1, size=len(gene)).astype(np.float32)
                mutated[mutation_mask] += noise[mutation_mask]

                # Clip values to stay within the right bounds
                mutated = bouncy_clip(mutated, 1)

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

    # def _blend_crossover(
    #     self, parent1: Genotype, parent2: Genotype, alpha: float = 0.5
    # ) -> tuple[Genotype, Genotype]:
    #     offspring1, offspring2 = [], []

    #     for gene1, gene2 in zip(parent1, parent2):
    #         # Calculate min and max values for each gene position
    #         min_vals = np.minimum(gene1, gene2)
    #         max_vals = np.maximum(gene1, gene2)

    #         # Calculate range and extend it by alpha
    #         gene_range = max_vals - min_vals
    #         extended_min = min_vals - alpha * gene_range
    #         extended_max = max_vals + alpha * gene_range

    #         # Generate random values within the extended range
    #         child1 = RNG.uniform(extended_min, extended_max).astype(np.float32)
    #         child2 = RNG.uniform(extended_min, extended_max).astype(np.float32)

    #         # Clip to maintain [0, 1] bounds
    #         child1 = np.clip(child1, 0.0, 1.0)
    #         child2 = np.clip(child2, 0.0, 1.0)

    #         offspring1.append(child1)
    #         offspring2.append(child2)

    #     return offspring1, offspring2


class BodyEA:
    def __init__(
        self,
        body_params: dict,
        mind_params: dict,
    ) -> None:
        """
        Simple evolutionary algorithm implementation.

        Args:
            body_params (dict): Parameters for the body evolution algorithm.
                Expected keys: population_size, generations,
                mutation_rate, crossover_rate, crossover_type, elitism,
                selection, tournament_size
            mind_params (dict): Parameters for the mind evolution algorithm.
                Expected keys: population_size, generations, mutation_rate,
                crossover_rate, crossover_type, elitism, selection, tournament_size
        """
        # Extract body parameters with defaults
        self.population_size = body_params.get("population_size", 20)
        self.generations = body_params.get("generations", 10)

        # Body EA specific parameters
        mutation_rate = body_params.get("mutation_rate", 0.1)
        crossover_type = body_params.get("crossover_type", "onepoint")
        self.crossover_rate = body_params.get("crossover_rate", 0.7)
        self.elitism = body_params.get("elitism", 1)
        self.selection = body_params.get("selection", "tournament")
        self.tournament_size = body_params.get("tournament_size", 3)

        # Initialize mutation and crossover operators
        self.mutate = Mutate(mutation_rate)
        self.crossover = Crossover(crossover_type)

        # Store mind parameters for use in _eval_func
        self.mind_params = mind_params

        # multi spawn top k
        self.top_k_re_eval = body_params.get("top_k_re_eval", 3)

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
        """
        phase 1: eval everyone in this generation on single spawn point
        phase 2: re eval only topp k on all terrains and aggregate
        """

        terrains = STOCHASTIC_SPAWN_POSITIONS

        base_spawn = self.current_spawn_point
        base_scores = [self._eval_func(ind, base_spawn) for ind in population]

        if self.top_k_re_eval == 0 or len(terrains) == 1:
            return base_scores

        k = min(self.top_k_re_eval, len(population))
        top_k_indices = list(np.argsort(base_scores))[-k:]
        adjusted_scores = base_scores.copy()

        for i in top_k_indices:
            per_terrain_score = []
            for spawn in terrains:
                if spawn == base_spawn:
                    per_terrain_score.append(base_scores[i])
                else:
                    per_terrain_score.append(self._eval_func(population[i], spawn))
            adjusted_scores[i] = np.mean(per_terrain_score)
        
        print(f"  → Re-evaluated top {k} individuals on {len(terrains)} terrains")

        return adjusted_scores

    def _eval_func(self, genotype: Genotype, spawn_point: list[float] | None = None) -> float:
        # Survives -> evaluate brain as before
        if spawn_point is None:
            spawn_point = self.current_spawn_point

        ea = MindEA(
            robot=Robot(spawn_point=spawn_point, body_genotype=genotype),
            population_size=self.mind_params.get("population_size", 10),
            generations=self.mind_params.get("generations", 1),
            mutation_rate=self.mind_params.get("mutation_rate", 0.5),
            crossover_rate=self.mind_params.get("crossover_rate", 0.0),
            crossover_type=self.mind_params.get("crossover_type", "onepoint"),
            elitism=self.mind_params.get("elitism", 2),
            selection=self.mind_params.get("selection", "tournament"),
            tournament_size=self.mind_params.get("tournament_size", 3),
        )
        _, _, avg_weights = ea.run()
        return calculate_slope_of_weights(avg_weights)




    def create_initial_population(self) -> list[Genotype]:        
        population = []

        with Pool() as pool:
            while len(population) < self.population_size:
                remaining = self.population_size - len(population)
                # Run evaluations in parallel
                results = pool.map(_evaluate_genotype, range(remaining))
                # Filter successful ones
                successful = [g for g in results if g is not None]
                population.extend(successful)
                print(len(population))

        print("Initial population created.")
        return population

    def apply_elitism(
        self, population: list[Genotype], fitness_scores: list[float]
    ) -> list[Genotype]:
        if self.elitism <= 0:
            return []

        # Get indices sorted by fitness (highest first)
        elite_indices = np.argsort(fitness_scores)[-self.elitism :]
        elite_individuals = [population[i] for i in elite_indices]

        return elite_individuals

    def _get_spawn_point(self, generation: int) -> list[float]:
        # Cycle through predefined spawn points based on generation number
        return STOCHASTIC_SPAWN_POSITIONS[0]
        return STOCHASTIC_SPAWN_POSITIONS[generation % len(STOCHASTIC_SPAWN_POSITIONS)]
    
    def load_population(self) -> list[Genotype]:
        with open("body_population.pkl", "rb") as f:
            return pkl.load(f)

    def export_population(self, population: list[Genotype], generation: int) -> None:
        with open(f"body_population_gen{generation}.pkl", "wb") as f:
            pkl.dump(population, f)

    def run(self, load_population: bool) -> tuple[Genotype, list[float], list[float]]:
        """
        Run the evolutionary algorithm.

        Returns:
            tuple: (best_individual, best_fitness_history, average_fitness_history)
        """
        # Initialize population
        if load_population:
            population = self.load_population()
            self.population_size = len(population)
        else:
            population = self.create_initial_population()

        # Track statistics
        best_fitness_history = []
        average_fitness_history = []

        for generation in range(self.generations):
            print(f"Generation {generation+1}/{self.generations}")

            self.current_spawn_point = self._get_spawn_point(generation)

            fitness_scores = self.evaluate_population(population)
            best_index = np.argmax(fitness_scores)
            best_individual = population[best_index]
            Robot(body_genotype=best_individual).save()

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
                offspring1, offspring2 = None, None
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

                if (
                    offspring1 is None
                    and is_robot_viable(Robot(body_genotype=child1))
                    and evaluate(
                        Robot(body_genotype=child1),
                        learn_test=True,
                        controller_func=controller.random
                    )
                ):
                    offspring1 = child1

                if (
                    offspring2 is None
                    and is_robot_viable(Robot(body_genotype=child2))
                    and evaluate(
                        Robot(body_genotype=child2),
                        learn_test=True,
                        controller_func=controller.random
                    )
                ):
                    offspring2 = child2

                if offspring1 is None or offspring2 is None:
                    continue  # Retry if either offspring is not viable

                # Add to new population (check size to avoid exceeding population_size)
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)

            # Update population
            population = new_population

            self.export_population(population, generation)
            # self.export_generation(generation)

        # Return best individual from final generation
        final_fitness_scores = self.evaluate_population(population)
        best_index = np.argmax(final_fitness_scores)
        best_individual = population[best_index]

        return best_individual, best_fitness_history, average_fitness_history
