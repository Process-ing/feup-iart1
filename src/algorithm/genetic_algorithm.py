from dataclasses import dataclass
import random
from typing import Iterator, List, Union, override
from src.model import Building
from src.model import RouterProblem
from src.algorithm.algorithm import Algorithm

@dataclass
class GeneticAlgorithmConfig:
    population_size: int
    init_routers: int
    max_similarity: float
    max_generations: Union[int, None]
    max_neighborhood: Union[int, None]

class GeneticAlgorithm(Algorithm):
    '''
    Genetic Algorithm
    '''
    def __init__(self, problem: RouterProblem, config: GeneticAlgorithmConfig) -> None:
        self.__problem = problem
        self.__config = config

    def placement_descent(self) -> Iterator[str]:
        init_routers = self.__config.init_routers
        max_neighborhood = self.__config.max_neighborhood

        for _ in range(init_routers):
            if self.__problem.building.get_num_uncovered_targets() == 0:
                break

            best_score = -1
            best_neighbor = None
            current_score = self.__problem.building.score

            num_neighbors = 0
            for operator in self.__problem.building.get_placement_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield 'No neighbor found'
                    continue

                if neighbor.score > current_score:
                    if neighbor.score > best_score:
                        best_score = neighbor.score
                        best_neighbor = neighbor

                    num_neighbors += 1
                    if num_neighbors == max_neighborhood:
                        yield f"{'Placed' if operator.place else 'Removed'} router at " \
                            f"({operator.row}, {operator.col})"
                        break

            if best_neighbor is not None:
                self.__problem.building = best_neighbor
            else:
                break

    def sort_population(self, population: list[Building]) -> None:
        population.sort(key=lambda individual: individual.score)

    def get_best_individual(self, population: list[Building]) -> Building:
        return max(population, key=lambda individual: individual.score)

    def crossover(self, population: List[Building], offspring: List[Building]) -> Iterator[str]:
        min_score = min(individual.score for individual in population)
        fitness_scores = [individual.score - min_score + 1 for individual in population]

        while len(offspring) < len(population):
            parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)

            children = parent1.crossover(parent2)
            if children is None:
                yield 'Crossover failed'
                continue

            offspring.extend(children)
            yield f'Crossover successful, offspring size: {len(offspring)}'

    def mutate(self, offspring: List[Building]) -> Iterator[str]:
        for i, child in enumerate(offspring):
            if random.random() < 0.5:
                for operator in child.get_neighborhood():
                    neighbor = operator.apply(child)
                    if neighbor is not None:
                        child = neighbor
                        yield 'Mutation successful'
                        break
                    yield 'Mutation failed'

            offspring[i] = child

    @override
    def run(self) -> Iterator[str]:
        max_generations = self.__config.max_generations
        population_size = self.__config.population_size
        max_similarity = self.__config.max_similarity

        original_building = self.__problem.building
        population = []
        best_score = -1
        best_neighbor = None

        for _ in range(population_size):
            self.__problem.building = original_building
            yield from self.placement_descent()

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

            # Check similarity of individuals
            filtered_offspring: List[Building] = []
            for i, child in enumerate(offspring):
                not_similar = True
                for j, individual in enumerate(population):
                    if child.is_similar(individual, max_similarity):
                        not_similar = False
                        break

                for j, other_child in enumerate(filtered_offspring):
                    if child.is_similar(other_child, max_similarity):
                        not_similar = False
                        break

                if not_similar:
                    filtered_offspring.append(child)

            # Replace the worst individuals in the population
            i = 0
            for j in range(len(filtered_offspring) - 1, -1, -1):
                if filtered_offspring[j].score < population[i].score:
                    break

                population[i] = filtered_offspring[j]
                i += 1
                yield 'Individual replaced'

            best_individual = self.get_best_individual([population[0], population[-1]])
            # Best individual is either the first or last in the population
            best_gen_score = best_individual.score

            if best_gen_score > best_score:
                best_score = population[0].score
                self.__problem.building = population[0]

            yield 'Best individual found'

    @staticmethod
    def get_default_init_routers(problem: RouterProblem) -> int:
        """Returns the maximum possible number of routers, according to the budget"""
        return problem.budget // (problem.router_price + problem.backbone_price)

