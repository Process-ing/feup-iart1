from dataclasses import dataclass
import random
from typing import Iterator, List, Optional, override
from src.algorithm.random_descent import RandomDescent, RandomDescentConfig
from src.model import Building
from src.model import RouterProblem
from src.algorithm.algorithm import Algorithm, AlgorithmConfig

@dataclass
class GeneticAlgorithmConfig(AlgorithmConfig):
    """
    Configuration class for the Genetic Algorithm.

    Attributes:
        population_size (int): The size of the population in the algorithm.
        init_routers (int): The initial number of routers to place in the problem.
        mutation_prob (float): The probability of mutation during the algorithm.
        max_generations (Optional[int]): The maximum number of generations to run the algorithm.
        max_neighborhood (Optional[int]): The maximum neighborhood size for placement descent.
        memetic (bool): Whether to perform a memetic phase after the genetic algorithm steps.
    """
    population_size: int
    init_routers: int
    mutation_prob: float
    max_generations: Optional[int]
    max_neighborhood: Optional[int]
    memetic: bool

    @classmethod
    def from_flags(cls, flags: dict[str, str],
                   default_init_routers: int) -> Optional['GeneticAlgorithmConfig']:
        """
        Creates a GeneticAlgorithmConfig instance from a dictionary of flags.

        Args:
            flags (dict): A dictionary where keys are configuration option names
            and values are their string representations.
            default_init_routers (int): The default number of routers if not specified in the flags.

        Returns:
            Optional[GeneticAlgorithmConfig]: An instance of GeneticAlgorithmConfig
            or None if flags are invalid.
        """
        if any(key not in ['population-size', 'init-routers', 'mutation-prob',
                   'max-generations', 'max-neighborhood', 'memetic'] for key in flags):
            return None

        try:
            population_size = int(flags['population-size']) if 'population-size' in flags else 10
            init_routers = int(flags['init-routers']) \
                if 'init-routers' in flags else default_init_routers
            mutation_prob = float(flags['mutation-prob']) if 'mutation-prob' in flags else 0.5
            max_generations = int(flags['max-generations']) if 'max-generations' in flags else None
            max_neighborhood = int(flags['max-neighborhood']) \
                if 'max-neighborhood' in flags else 5
            memetic = bool(flags['memetic']) if 'memetic' in flags else False
        except ValueError:
            return None

        return cls(population_size, init_routers, mutation_prob,
                   max_generations, max_neighborhood, memetic)

class GeneticAlgorithm(Algorithm):
    """
    Genetic Algorithm for solving router placement problems.

    This class uses a genetic algorithm to optimize router placements in a building
    to maximize coverage
    and minimize costs, while considering constraints like available budget and placement rules.
    """

    def __init__(self, problem: RouterProblem, config: GeneticAlgorithmConfig) -> None:
        """
        Initializes the GeneticAlgorithm instance with the given problem and configuration.

        Args:
            problem (RouterProblem): The problem instance containing the current
            building and constraints.
            config (GeneticAlgorithmConfig): The configuration parameters for the
            genetic algorithm.
        """
        self.__problem = problem
        self.__config = config

    def placement_descent(self) -> Iterator[Optional[str]]:
        """
        Performs a greedy descent algorithm to place routers in the building,
        optimizing for coverage.

        Yields:
            Optional[str]: A message indicating the placement or removal of a router,
            or `None` if no improvement is made.
        """
        init_routers = self.__config.init_routers
        max_neighborhood = self.__config.max_neighborhood

        for _ in range(init_routers):
            if self.__problem.building.get_num_uncovered_targets() == 0:
                break
            if self.__problem.get_available_budget(self.__problem.building) \
                <= self.__problem.router_price + self.__problem.backbone_price:
                break

            best_score = -1
            best_neighbor = None
            current_score = self.__problem.building.score

            neighbor_count = 0
            for operator in self.__problem.building.get_placement_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield None
                    continue

                if neighbor.score > current_score:
                    if neighbor.score > best_score:
                        best_score = neighbor.score
                        best_neighbor = neighbor

                    neighbor_count += 1
                    if neighbor_count == max_neighborhood:
                        yield f"{'Placed' if operator.place else 'Removed'} router at " \
                            f"({operator.row}, {operator.col})"
                        break
                yield None

            if best_neighbor is not None:
                self.__problem.building = best_neighbor
            else:
                break

    def sort_population(self, population: List[Building], reverse: bool = False) -> None:
        """
        Sorts a population of buildings by their score.

        Args:
            population (List[Building]): The population of buildings to be sorted.
            reverse (bool): Whether to sort in reverse order (from highest to lowest
            score). Defaults to False.
        """
        population.sort(key=lambda individual: individual.score, reverse=reverse)

    def get_best_individual(self, population: List[Building]) -> Building:
        """
        Returns the building with the highest score from a population.

        Args:
            population (List[Building]): The population of buildings.

        Returns:
            Building: The building with the highest score in the population.
        """
        return max(population, key=lambda individual: individual.score)

    def crossover(self, population: List[Building],
                  offspring: List[Building]) -> Iterator[Optional[str]]:
        """
        Performs crossover between parent buildings to generate offspring.

        Args:
            population (List[Building]): The population of parent buildings.
            offspring (List[Building]): The list to store the generated offspring.

        Yields:
            Optional[str]: A message indicating whether crossover was successful or failed.
        """
        min_score = min(individual.score for individual in population)
        fitness_scores = [individual.score - min_score + 1 for individual in population]

        while len(offspring) < len(population):
            parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)

            children = parent1.crossover(parent2)
            if children is None:
                yield f'Crossover failed, offspring size: {len(offspring)}'
                continue

            offspring.extend(children)
            yield f'Crossover successful, offspring size: {len(offspring)}'

    def mutate(self, offspring: List[Building]) -> Iterator[Optional[str]]:
        """
        Mutates the offspring buildings based on the mutation probability.

        Args:
            offspring (List[Building]): The list of offspring to mutate.

        Yields:
            Optional[str]: A message indicating if a mutation has occurred or `None`
            if no mutation took place.
        """
        mutation_prob = self.__config.mutation_prob

        for i, child in enumerate(offspring):
            if random.random() < mutation_prob:
                for operator in child.get_neighborhood():
                    neighbor = operator.apply(child)
                    if neighbor is not None:
                        child = neighbor
                        yield 'Mutated child'
                        break
                    yield None

            offspring[i] = child

    def deletion(self, population: List[Building],
                 offspring: List[Building]) -> Iterator[Optional[str]]:
        """
        Replaces the worst individuals in the population with new, non-similar offspring.

        Args:
            population (List[Building]): The current population of buildings.
            offspring (List[Building]): The list of offspring to be evaluated
            and potentially added.

        Yields:
            Optional[str]: A message indicating the result of the deletion process.
        """
        population_size = self.__config.population_size

        # Check similarity of individuals
        filtered_offspring: List[Building] = []
        for child in offspring:
            not_similar = True
            for individual in population:
                if child.is_same(individual):
                    not_similar = False
                    break

            for other_child in filtered_offspring:
                if child.is_same(other_child):
                    not_similar = False
                    break

            if not_similar:
                filtered_offspring.append(child)
            yield None

        # Replace the worst individuals in the population
        population.extend(filtered_offspring)
        self.sort_population(population, reverse=True)
        del population[population_size:]

    def memetic_phase(self) -> Iterator[Optional[str]]:
        """
        Performs a random descent optimization phase as a memetic phase after
        the genetic algorithm steps.

        Yields:
            Optional[str]: A message indicating the start of the memetic phase
            and its progress.
        """
        yield 'memetic phase started'
        random_descent = RandomDescent(self.__problem, RandomDescentConfig(
            max_neighborhood=self.__config.max_neighborhood,
            max_iterations=None
        ))

        yield from random_descent.run()

    @override
    def run(self) -> Iterator[Optional[str]]:
        """
        Executes the full genetic algorithm process, including placement
        descent, crossover, mutation, deletion, 
        and optional memetic phase.

        Yields:
            Optional[str]: Messages indicating the progress of the algorithm,
            such as population generation, 
            crossover success, and best individual updates.
        """
        max_generations = self.__config.max_generations
        population_size = self.__config.population_size
        memetic = self.__config.memetic

        original_building = self.__problem.building
        population = []
        best_score = -1
        best_neighbor = None

        yield 'Generating initial population'
        for _ in range(population_size):
            self.__problem.building = original_building
            for _ in self.placement_descent():
                yield None

            population.append(self.__problem.building)
            score = self.__problem.building.score
            if score > best_score:
                best_score = score
                best_neighbor = self.__problem.building

        assert best_neighbor is not None  # Can only happen if population size is 0
        self.__problem.building = best_neighbor

        generation_iter = range(max_generations) if max_generations is not None else iter(int, 1)

        for _ in generation_iter:
            offspring: List[Building] = []
            yield from self.crossover(population, offspring)
            yield from self.mutate(population)

            yield from self.deletion(population, offspring)

            best_individual = population[0]  # Population comes out sorted in decreasing order
            # Best individual is either the first or last in the population
            best_gen_score = best_individual.score

            if best_gen_score > best_score:
                best_score = population[0].score
                self.__problem.building = population[0]

            yield 'Best individual found'

        if memetic:
            yield from self.memetic_phase()

    @staticmethod
    def get_default_init_routers(problem: RouterProblem) -> int:
        """
        Returns the maximum possible number of routers that can be placed
        according to the available budget.

        Args:
            problem (RouterProblem): The problem instance containing the
            budget and router price information.

        Returns:
            int: The maximum number of routers that can be placed within the budget.
        """
        return problem.budget // (problem.router_price + problem.backbone_price)
