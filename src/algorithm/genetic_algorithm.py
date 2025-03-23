from copy import deepcopy
import random
from src.model import RouterProblem
from src.algorithm.algorithm import Algorithm


class GeneticAlgorithm(Algorithm):
    """
    Genetic Algorithm
    """
    def __init__(self, problem: RouterProblem, population_size: int = 10, max_generations: int = 10) -> None:
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

        # Perform crossover to create offspring
        offspring = []
        for _ in range(self.__population_size // 2):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = parent1.crossover(parent2)
            offspring.extend([child1, child2])

        # Perform mutation on offspring
        for individual in offspring:
            if random.random() < 0.1:  # Mutation probability
                mutation_point = random.randint(0, len(individual) - 1)
                individual[mutation_point] = self.__problem.mutate(individual[mutation_point])

        # Replace old population with new offspring
        self.__population = offspring

        # Check termination condition
        self.__done = self.__max_generations == 0
        self.__max_generations -= 1