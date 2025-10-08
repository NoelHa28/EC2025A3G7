# --- robot preview bits (MuJoCo) ---
import os
import mujoco as mj
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker
import controller as ctrl_module
from simulate import experiment
from opposites import has_core_opposite_pair, simple_symmetry_score, worm_score


# enable/disable viewer via env vars (no code edits later)
SHOW_ROBOT      = os.getenv("SHOW_ROBOT", "0") == "1"   # 1 -> open 3D viewer
SHOW_ONLY_KILLS = os.getenv("SHOW_ONLY_KILLS", "1") == "1"
PREVIEW_SECS    = float(os.getenv("PREVIEW_SECS", "2.0"))

def _preview_robot(robot, seconds: float = PREVIEW_SECS) -> None:
    """Open a short MuJoCo viewer session to inspect the morphology."""
    mj.set_mjcb_control(None)
    core = construct_mjspec_from_graph(robot.graph)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=ctrl_module.cpg, tracker=tracker)
    experiment(robot=robot, core=core, controller=ctrl, mode="launcher", duration=seconds)

# --- symmetry / eval imports ---
from opposites import has_core_opposite_pair, simple_symmetry_score

KILL_FITNESS = -100.0  # sentinel filtered out by selection

SHOW_MORPH = os.getenv("SHOW_MORPH", "0") == "1"        # set to 1 to show popups
SHOW_ONLY_KILLS = os.getenv("SHOW_ONLY_KILLS", "1") == "1"  # default: show only killed ones

import multi_spawn as ms
import dynamic_duration as dd
from .mind import MindEA
from typing import Any
from collections.abc import Callable
import multiprocessing as mp
import numpy as np
from robot import Robot
from morphology_constraints import is_robot_viable
from ariel import console

SEED = 42
RNG = np.random.default_rng(SEED)
KILL_FITNESS = -100.0  # you already filter this out in selection


type Genotype = list[np.ndarray]

def clip_values(genotype: Genotype) -> Genotype:
    """Clip genotype values to stay within [0, 1] range."""
    genotype = np.abs(genotype) % 2
    genotype[genotype > 1] = 2 - genotype[genotype > 1]
    return genotype

class Mutate:
    def __init__(self, mutation_rate: float = 0.1) -> None:
        self.mutation_rate = mutation_rate

    def gaussian(self, genotype: Genotype) -> Genotype:
        mutated_genotype = []
        # Set different sigmas per genotype
        sigmas = [0.05, 0.1, 0.1]

        for gene, sigma in zip(genotype, sigmas):
            # Create a copy to avoid modifying the original
            mutated = gene.copy()
            # Create a mask for which genes to mutate
            mutation_mask = RNG.random(len(gene)) < self.mutation_rate

            # Apply Gaussian noise to selected genes
            if np.any(mutation_mask):
                noise = RNG.normal(0, sigma, size=len(gene)).astype(np.float32)
                mutated[mutation_mask] += noise[mutation_mask]

                # Change values to stay within [0, 1] range
                mutated = clip_values(mutated)

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
            child1 = clip_values(child1)
            child2 = clip_values(child2)

            offspring1.append(child1)
            offspring2.append(child2)

        return offspring1, offspring2


class BodyEA:
    def __init__(
        self,
        body_params: dict,
        mind_params: dict,
        evaluator: Callable[[Any], float] = None,
    ) -> None:
        """
        Simple evolutionary algorithm implementation.

        Args:
            body_params (dict): Parameters for the body evolution algorithm.
                Expected keys: population_size, generations, genotype_size,
                mutation_rate, crossover_rate, crossover_type, elitism,
                selection, tournament_size
            mind_params (dict): Parameters for the mind evolution algorithm.
                Expected keys: population_size, generations, mutation_rate,
                crossover_rate, crossover_type, elitism, selection, tournament_size
            evaluator (callable): Function to evaluate the fitness of an individual.
        """
        # Extract body parameters with defaults
        self.population_size = body_params.get("population_size", 20)
        self.generations = body_params.get("generations", 10)
        self.genotype_size = body_params.get("genotype_size", 100)
        self.evaluator = evaluator  # Function to evaluate fitness

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

        self.best_so_far: float | None = None  # Track best fitness so far
        self.dynamic_duration_enabled: bool = body_params.get(
            "dynamic_duration", True
        )  # expose a knob

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
        robot = Robot(genotype)

        # Use mind_params for MindEA configuration
        try:
            robot = Robot(genotype)
        except RuntimeError:
            return KILL_FITNESS

        G = robot.graph

        # --- worm exception (keeps long single-tail bodies) ---
        w_score, w_len, w_branches = worm_score(G)
        WORM_LEN_MIN = 8          # tweak if needed
        WORM_KEEP_THRESH = 0.65   # 0..1 chaininess
        worm_ok = (w_len >= WORM_LEN_MIN) and (w_score >= WORM_KEEP_THRESH)

        # cheap hard gate unless it's a worm
        if not has_core_opposite_pair(G) and not worm_ok:
            print("KILL (no opposite pair on core and not worm-like)")
            return KILL_FITNESS

        # whole-body symmetry
        sym = simple_symmetry_score(G, max_depth=3)

        # base kill prob from symmetry
        threshold, softness = 0.6, 0.3
        p_kill = max(0.0, min(1.0, (threshold - sym) / max(softness, 1e-6)))

        # soften kill if worm-like (or bypass entirely)
        if worm_ok:
            p_kill *= 0.2  # 80% reduction; or set to 0 to always keep worms

        # decide + optional preview
        kill = RNG.random() < p_kill
        print(
            f"sym={sym:.3f}  p_kill={p_kill:.2f}  worm_score={w_score:.2f} "
            f"len={w_len} branches={w_branches}  decision={'KILL' if kill else 'KEEP'}"
        )
        if SHOW_ROBOT and ((not SHOW_ONLY_KILLS) or kill):
            _preview_robot(robot, seconds=PREVIEW_SECS)

        if kill:
            return KILL_FITNESS

        # survives -> MindEA
        ea = MindEA(
            robot=robot,
            population_size=self.mind_params.get("population_size", 10),
            generations=self.mind_params.get("generations", 1),
            mutation_rate=self.mind_params.get("mutation_rate", 0.5),
            crossover_rate=self.mind_params.get("crossover_rate", 0.0),
            crossover_type=self.mind_params.get("crossover_type", "onepoint"),
            elitism=self.mind_params.get("elitism", 2),
            selection=self.mind_params.get("selection", "tournament"),
            tournament_size=self.mind_params.get("tournament_size", 3),
        )
        _, weights, _ = ea.run()
        return float(max(weights))


    def create_initial_population(self) -> list[Genotype]:
        population = []
        killed = 0
        while len(population) < self.population_size:
            genotype = self.random_genotype()
            if is_robot_viable(Robot(genotype)):
                population.append(genotype)
            else:
                killed += 1

        console.log(
            f"Initial population: {self.population_size} viable, {killed} killed"
        )
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
            # choose and store the generation’s spawn (respects MULTISPAWN_ENABLED)
            ms.set_current_spawn(ms.MULTISPAWN_ENABLED)

            # set duration for THIS generation using best_so_far (from previous gens)
            dur = dd.update_for_generation(self.best_so_far, ms.CURRENT_SPAWN)
            print(f"[GEN {generation+1}] spawn={ms.CURRENT_SPAWN}, duration={dur}s")

            # Evaluate population
            fitness_scores = self.evaluate_population(population)

            # Track statistics
            best_fitness = max(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness_history.append(best_fitness)
            average_fitness_history.append(average_fitness)
            self.best_so_far = (
                best_fitness
                if (self.best_so_far is None)
                else max(self.best_so_far, best_fitness)
            )
            # Print progress
            if generation % 10 == 0 or generation == self.generations - 1:
                print(
                    f"Generation {generation}: Best={best_fitness:.4f}, Avg={average_fitness:.4f}"
                )

            # Apply elitism (preserve best individuals)
            elite_individuals = self.apply_elitism(population, fitness_scores)

            # Create new population
            new_population = elite_individuals.copy()

            child1, child2 = None, None

            # Fill rest of population with offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitness_scores)

                # Apply crossover
                if RNG.random() < self.crossover_rate:
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                else:
                    # If no crossover, children are copies of parents
                    offspring1, offspring2 = [gene.copy() for gene in parent1], [
                        gene.copy() for gene in parent2
                    ]

                # Apply mutation
                offspring1 = self.mutate.gaussian(offspring1)
                offspring2 = self.mutate.gaussian(offspring2)

                if child1 is None and is_robot_viable(Robot(offspring1)):
                    child1 = offspring1

                if child2 is None and is_robot_viable(Robot(offspring2)):
                    child2 = offspring2

                if child1 is None or child2 is None:
                    continue  # Retry if either child is not viable

                # Add to new population (check size to avoid exceeding population_size)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

                child1, child2 = None, None  # Reset for next pair

            # Update population
            population = new_population

        # Return best individual from final generation
        final_fitness_scores = self.evaluate_population(population)
        best_index = np.argmax(final_fitness_scores)
        best_individual = population[best_index]

        return best_individual, best_fitness_history, average_fitness_history
