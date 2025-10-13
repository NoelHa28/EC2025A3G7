import multiprocessing as mp
import numpy as np
from robot import Robot
from evaluate import evaluate

CTX = mp.get_context("spawn")
CPU_COUNT = mp.cpu_count()

from consts import RNG

ALL_TIME_BEST_FITNESS = -100

GenotypeWithSigma = tuple[np.ndarray, np.ndarray]  # (genotype, sigma)

class SelfAdaptiveMutation:
    """Self-adaptive mutation (per-gene sigma) for evolutionary strategies."""
    def __init__(self, n_genes: int, init_sigma: float = 0.05, tau: float | None = None):
        self.n_genes = n_genes
        self.init_sigma = init_sigma
        self.tau = tau or 1 / np.sqrt(2 * np.sqrt(n_genes))

    def __call__(self, individual: GenotypeWithSigma) -> GenotypeWithSigma:
        genotype, sigma = individual

        # Update sigmas: log-normal self-adaptation
        global_factor = np.exp(self.tau * RNG.normal())
        per_gene_factor = np.exp(self.tau * RNG.normal(size=self.n_genes))
        sigma_new = sigma * global_factor * per_gene_factor
        sigma_new = np.clip(sigma_new, 1e-5, 0.5)

        # Mutate genotype
        mutation = sigma_new * RNG.normal(size=self.n_genes)
        genotype_new = np.clip(genotype + mutation, 0.0, 1.0)

        return genotype_new, sigma_new

class Crossover:
    """Crossover operators for genotypes."""
    def __init__(self, crossover_type: str):
        self.crossover_type = crossover_type

    def __call__(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.crossover_type == "onepoint":
            return self.one_point(parent1, parent2)
        elif self.crossover_type == "uniform":
            return self.uniform(parent1, parent2)
        elif self.crossover_type == "blend":
            return self.blend(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}")

    def one_point(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point = RNG.integers(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1.astype(np.float32), child2.astype(np.float32)

    def uniform(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = RNG.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1.astype(np.float32), child2.astype(np.float32)

    def blend(self, parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        child1 = (alpha * parent1 + (1 - alpha) * parent2).astype(np.float32)
        child2 = (alpha * parent2 + (1 - alpha) * parent1).astype(np.float32)
        return child1, child2

class MindEA:
    def __init__(
        self,
        robot: Robot,
        population_size: int,
        generations: int,
        crossover_rate: float = 0.7,
        crossover_type: str = "onepoint",
        elitism: int = 1,
        selection: str = "tournament",
        tournament_size: int = 3,
    ) -> None:
        self.robot = robot
        self.population_size = population_size
        self.generations = generations
        self.mutate = SelfAdaptiveMutation(self.robot.brain.generate_random_cpg_genotype(self.robot._number_of_hinges).size)
        self.crossover = Crossover(crossover_type)
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.selection = selection
        self.tournament_size = tournament_size

    def random_individual(self) -> GenotypeWithSigma:
        genotype = self.robot.brain.generate_random_cpg_genotype(self.robot._number_of_hinges)
        sigma = np.full(genotype.shape, 0.05, dtype=np.float32)
        return genotype, sigma

    def create_initial_population(self) -> list[GenotypeWithSigma]:
        return [self.random_individual() for _ in range(self.population_size)]

    def select_parents(self, population: list[GenotypeWithSigma], fitness_scores: list[float]) -> tuple[GenotypeWithSigma, GenotypeWithSigma]:
        viable_indices = [i for i, f in enumerate(fitness_scores) if f != -100.0]
        if len(viable_indices) < 2:
            viable_indices = list(range(len(population)))

        viable_population = [population[i] for i in viable_indices]
        viable_fitness_scores = [fitness_scores[i] for i in viable_indices]

        if self.selection == "tournament":
            return self._tournament_selection(viable_population, viable_fitness_scores), \
                   self._tournament_selection(viable_population, viable_fitness_scores)
        elif self.selection == "roulette":
            return self._roulette_selection(viable_population, viable_fitness_scores), \
                   self._roulette_selection(viable_population, viable_fitness_scores)
        else:
            raise ValueError(f"Unknown selection method: {self.selection}")

    def _tournament_selection(self, population: list[GenotypeWithSigma], fitness_scores: list[float]) -> GenotypeWithSigma:
        idxs = RNG.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = idxs[0]
        best_score = fitness_scores[best_idx]
        for idx in idxs[1:]:
            if fitness_scores[idx] > best_score:
                best_idx = idx
                best_score = fitness_scores[idx]
        return population[best_idx]

    def _roulette_selection(self, population: list[GenotypeWithSigma], fitness_scores: list[float]) -> GenotypeWithSigma:
        sorted_idx = np.argsort(fitness_scores)
        ranks = np.zeros(len(fitness_scores))
        for i, idx in enumerate(sorted_idx):
            ranks[idx] = i + 1
        probs = ranks / ranks.sum()
        selected_idx = RNG.choice(len(population), p=probs)
        return population[selected_idx]

    def evaluate_population(self, population: list[GenotypeWithSigma]) -> list[float]:
        with CTX.Pool(CPU_COUNT) as pool:
            return pool.map(self._eval_func, population)

    def _eval_func(self, individual: GenotypeWithSigma) -> float:
        genotype, _ = individual
        self.robot.brain.set_genotype(genotype)
        return evaluate(self.robot)

    def apply_elitism(self, population: list[GenotypeWithSigma], fitness_scores: list[float]) -> list[GenotypeWithSigma]:
        if self.elitism <= 0:
            return []
        elite_indices = np.argsort(fitness_scores)[-self.elitism:]
        return [population[i] for i in elite_indices]

    def run(self) -> tuple[GenotypeWithSigma, list[float], list[float]]:
        population = self.create_initial_population()
        best_history, avg_history = [], []
        global ALL_TIME_BEST_FITNESS
        for gen in range(self.generations):
            fitness_scores = self.evaluate_population(population)

            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            best_history.append(best_fitness)
            avg_history.append(avg_fitness)

            if best_fitness > ALL_TIME_BEST_FITNESS:
                ALL_TIME_BEST_FITNESS = best_fitness
                self.robot.brain.set_genotype(population[np.argmax(fitness_scores)][0])
                self.robot.save()
                print("  → New all-time best fitness:", ALL_TIME_BEST_FITNESS)

            print(f"Gen {gen+1}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")

            elite = self.apply_elitism(population, fitness_scores)
            new_pop = elite.copy()

            while len(new_pop) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness_scores)

                # Crossover on genotype only
                child1_gen, child2_gen = parent1[0].copy(), parent2[0].copy()
                if RNG.random() < self.crossover_rate:
                    child1_gen, child2_gen = self.crossover(parent1[0], parent2[0])

                child1 = (child1_gen, parent1[1].copy())
                child2 = (child2_gen, parent2[1].copy())

                # Self-adaptive mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_pop.append(child1)
                if len(new_pop) < self.population_size:
                    new_pop.append(child2)

            population = new_pop

        # Return best individual
        final_scores = self.evaluate_population(population)
        best_idx = np.argmax(final_scores)
        return population[best_idx][0], best_history, avg_history