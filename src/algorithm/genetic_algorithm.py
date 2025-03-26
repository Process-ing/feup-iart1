from copy import deepcopy
import random
from typing import Iterator, override
from src.model import Building, Operator
from src.model import RouterProblem
from src.algorithm.algorithm import Algorithm
from src.algorithm.random_descent import RandomDescent


class GeneticAlgorithm(Algorithm):
    """
    Genetic Algorithm
    """
    def __init__(self, problem: RouterProblem, population_size: int = 10, initial_routers: int = 30, max_generations: int = 1000000) -> None:
        self.__problem = problem
        self.__population_size = population_size
        self.__max_generations = max_generations
        self.__initial_routers = initial_routers

    def placement_descent(self) -> Iterator[None]:
        found_neighbor = False
        current_score = self.__problem.get_score(self.__problem.building)

        for _ in range(self.__initial_routers):
            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield
                    continue

                if self.__problem.get_score(neighbor) > current_score:
                    self.__problem.building = neighbor
                    found_neighbor = True
                    break
                yield

            yield
            if not found_neighbor:
                break

    def sort_population(self, population: list[Building]) -> None:
        population.sort(key=lambda individual: self.__problem.get_score(individual))

    @override
    def run(self) -> Iterator[None]:
        original_building = self.__problem.building
        population = []
        best_score = float('-inf')
        best_neighbor = None

        for _ in range(self.__population_size):
            self.__problem.building = original_building
            yield from self.placement_descent()

            population.append(self.__problem.building)
            score = self.__problem.get_score(self.__problem.building)
            if score > best_score:
                best_score = score
                best_neighbor = self.__problem.building

        self.__problem.building = best_neighbor

        for _ in range(self.__max_generations):
            # Evaluate fitness of the population
            fitness_scores = [self.__problem.get_score(individual) for individual in population]

            # Perform crossover to create offspring
            offspring: list[Building] = []
            while len(offspring) < self.__population_size:
                parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)

                children = parent1.crossover(parent2)
                if children is None:
                    yield
                    continue

                children = list(children)
                for i, child in enumerate(children):
                    child_score = self.__problem.get_score(child)

                    for _ in range(self.__initial_routers):
                        operator = next(child.get_neighborhood())
                        neighbor = operator.apply(child)
                        if not neighbor:
                            yield
                            continue

                        neighbor_score = self.__problem.get_score(neighbor)
                        if neighbor_score > child_score:
                            children[i] = neighbor
                            child_score = neighbor_score
                            yield

                offspring.extend(children)
                yield

            self.sort_population(population)
            self.sort_population(offspring)

            for i in range(len(population)):
                if self.__problem.get_score(offspring[i]) < self.__problem.get_score(population[-i]):
                    break

                population[-i] = offspring[i]
                yield

            print("", max(self.__problem.get_score(individual) for individual in population))
            best_offspring_score = self.__problem.get_score(offspring[0])
            if best_offspring_score > best_score:
                best_score = self.__problem.get_score(offspring[0])
                self.__problem.building = offspring[0]

            yield
