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
    def __init__(self, problem, temperature = 100000, cooling_schedule = 1) -> None:
        self.__problem = problem
        self.__temperature = temperature
        self.__cooling_schedule = cooling_schedule

    def step(self):
        current_score = self.__problem.get_score(self.__problem.building)

        for operator in self.__problem.building.get_neighborhood():
            neighbor = operator.apply(self.__problem.building)
            if not neighbor:
                continue

            neighbor_score = self.__problem.get_score(neighbor)

            if neighbor_score > current_score:
                self.__problem.building = neighbor
                break
            else:
                probability = math.exp(float(neighbor_score - current_score) / self.__temperature)
                if random.random() < probability:
                    self.__problem.building = neighbor
                    break

        self.__temperature *= self.__cooling_schedule

    def done(self) -> bool:
        return self.__done