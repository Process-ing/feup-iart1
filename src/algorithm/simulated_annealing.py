from typing import Iterator, override
from src.model.problem import RouterProblem
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
    def __init__(self, problem: RouterProblem, temperature: float = 100.0,
                 cooling_schedule: float = 0.99, max_iterations: int | None = None) -> None:
        self.__problem = problem
        self.__max_iterations = max_iterations
        self.__temperature = temperature
        self.__cooling_schedule = cooling_schedule

    @override
    def run(self) -> Iterator[None]:
        for _ in range(self.__max_iterations) if self.__max_iterations is not None else iter(int, 1):
            current_score = self.__problem.building.score

            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield
                    continue

                neighbor_score = neighbor.score

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
