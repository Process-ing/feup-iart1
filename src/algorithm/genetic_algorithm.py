from copy import deepcopy
import random
from typing import override
from src.model import Building
from src.model import RouterProblem
from src.algorithm.algorithm import Algorithm


class GeneticAlgorithm(Algorithm):
    """
    Genetic Algorithm
    """
    def __init__(self, problem: RouterProblem, population_size: int = 10, max_generations: int = 1000) -> None:
        self.__problem = problem
        self.__population_size = population_size
        self.__max_generations = max_generations
        self.__population = [deepcopy(problem.building) for _ in range(population_size)]
        self.__done = False

    def step(self):
        # Evaluate fitness of the population
        fitness_scores = [self.__problem.get_score(individual) for individual in self.__population]

        # Select parents based on fitness (e.g., roulette wheel selection)
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = [
            self.__population[i]
            for i in range(self.__population_size)
            if random.random() < probabilities[i]
        ]
        if len(parents) < 2:
            parents = self.__population

        # Perform crossover to create offspring
        offspring: list[Building] = []
        for _ in range(self.__population_size // 2):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = parent1.crossover(parent2)
            offspring.extend([child1, child2])

        # Perform mutation on offspring
        for i, individual in enumerate(offspring):
            if random.random() < 0.5:  # Mutation probability
                for operator in individual.get_neighborhood():
                    mutated = operator.apply(individual)
                    if mutated is not None and self.__problem.get_score(mutated) > self.__problem.get_score(individual):
                        offspring[i] = mutated
                        break

        best_score = self.__problem.get_score(self.__problem.building)
        for individual in offspring:
            score = self.__problem.get_score(individual)
            if score > best_score:
                best_score = score
                self.__problem.building = individual

        # Replace old population with new offspring
        self.__population = offspring

        # Check termination condition
        self.__done = self.__max_generations == 0
        self.__max_generations -= 1

    @override
    @property
    def done(self):
        return self.__done