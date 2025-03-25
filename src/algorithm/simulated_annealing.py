from typing import override
from src.algorithm.algorithm import Algorithm
import math
import random

class SimulatedAnnealing(Algorithm):
    """
    Simulated Annealing Algorithm
    Picks a random neighbor to explore
      - If the neighbor is better, choose it
      - Else only accept with a certain probability (based on a temperature and cooling schedule)
    Always decrease the temperature (based on cooling schedule)
    """
    def __init__(self, problem, max_iterations: int | None = None, temperature = 100000, cooling_schedule = 0.99) -> None:
        self.__problem = problem
        self.__max_iterations = max_iterations
        self.__temperature = temperature
        self.__cooling_schedule = cooling_schedule

    @override
    def run(self):
        for _ in range(self.__max_iterations) if self.__max_iterations is not None else iter(int, 1):
            current_score = self.__problem.get_score(self.__problem.building)

            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield
                    continue

                neighbor_score = self.__problem.get_score(neighbor)

                if neighbor_score > current_score:
                    self.__problem.building = neighbor
                    yield
                    break
                else:
                    probability = math.exp(float(neighbor_score - current_score) / self.__temperature)
                    if random.random() < probability:
                        self.__problem.building = neighbor
                        yield
                        break
                    else:
                        yield

            self.__temperature *= self.__cooling_schedule
