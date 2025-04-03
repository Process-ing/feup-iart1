from dataclasses import dataclass
from typing import Iterator, Union, override
import math
import random

from src.model.problem import RouterProblem
from src.algorithm.algorithm import Algorithm

@dataclass
class SimulatedAnnealingConfig:
    max_iterations: Union[int, None]
    init_temperature: float
    cooling_schedule: float

class SimulatedAnnealing(Algorithm):
    '''
    Simulated Annealing Algorithm
    Picks a random neighbor to explore
      - If the neighbor is better, choose it
      - Else only accept with a certain probability (based on a temperature and cooling schedule)
    Always decrease the temperature (based on cooling schedule)
    '''
    def __init__(self, problem: RouterProblem, config: SimulatedAnnealingConfig) -> None:
        self.__problem = problem
        self.__config = config

    @override
    def run(self) -> Iterator[str]:
        max_iterations = self.__config.max_iterations
        temperature = self.__config.init_temperature
        cooling_schedule = self.__config.cooling_schedule

        round_iter = range(max_iterations) if max_iterations is not None else iter(int, 1)
        for _ in round_iter:
            current_score = self.__problem.building.score

            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield 'No neighbor found'
                    continue

                neighbor_score = neighbor.score

                if neighbor_score > current_score:
                    self.__problem.building = neighbor
                    action = 'Placed' if operator.place else 'Removed'
                    yield f'{action} router at ({operator.row}, {operator.col})'
                    break

                probability = math.exp(float(neighbor_score - current_score) / temperature)
                if random.random() < probability:
                    self.__problem.building = neighbor
                    action = 'Placed' if operator.place else 'Removed'
                    position = f'({operator.row}, {operator.col})'
                    yield f'{action} router at {position}'
                    break
                yield 'Rejected worse neighbor'

            temperature *= cooling_schedule
