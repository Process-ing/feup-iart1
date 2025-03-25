from copy import deepcopy
import random
from typing import Iterator, override
from src.model import Building, Operator
from src.model import RouterProblem
from src.algorithm.algorithm import Algorithm


class GeneticAlgorithm(Algorithm):
    """
    Genetic Algorithm
    """
    def __init__(self, problem: RouterProblem, population_size: int = 10, initial_routers: int = 200, max_generations: int = 1000) -> None:
        self.__problem = problem
        self.__population_size = population_size
        self.__max_generations = max_generations
        self.__initial_routers = initial_routers

    @override
    def run(self) -> Iterator[None]:
        original_building = self.__problem.building
        population = []
        best_score = float('-inf')
        best_neighbor = None

        for _ in range(self.__population_size):
            self.__problem.building = original_building

            neighborhood = self.__problem.building.get_target_cells()
            random.shuffle(neighborhood)
            neighborhood = neighborhood

            for row, col in neighborhood:
                operator = Operator(True, row, col, self.__problem.check_budget)

                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield
                    break

                self.__problem.building = neighbor
                yield

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
            num_crosses = 0
            while num_crosses < self.__population_size // 2:
                parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)

                children = parent1.crossover(parent2)
                if children is None:
                    yield
                    continue

                offspring.extend(children)
                num_crosses += 1
                yield

            # Perform mutation on offspring
            for i, individual in enumerate(offspring):
                for operator in individual.get_neighborhood():
                    mutated = operator.apply(individual)
                    if mutated is not None and self.__problem.get_score(mutated) > self.__problem.get_score(individual):
                        offspring[i] = mutated
                        yield
                        break

            best_score = self.__problem.get_score(self.__problem.building)
            for individual in offspring:
                score = self.__problem.get_score(individual)
                if score > best_score:
                    best_score = score
                    self.__problem.building = individual
            yield

            # Replace old population with new offspring
            population = offspring
