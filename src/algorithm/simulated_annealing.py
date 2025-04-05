from dataclasses import dataclass
from typing import Iterator, Optional, override
import math
import random

from src.model.problem import RouterProblem
from src.algorithm.algorithm import Algorithm, AlgorithmConfig

@dataclass
class SimulatedAnnealingConfig(AlgorithmConfig):
    max_iterations: Optional[int]
    init_temperature: float
    cooling_schedule: float

    @classmethod
    def from_flags(cls, flags: dict[str, str]) -> Optional['SimulatedAnnealingConfig']:
        try:
            max_iterations = int(flags['max-iterations']) if 'max-iterations' in flags else None
            init_temperature = float(flags['init-temperature']) \
                if 'init-temperature' in flags else 1000.0
            cooling_schedule = float(flags['cooling-schedule']) \
                if 'cooling-schedule' in flags else 0.99
        except ValueError:
            return None

        return cls(max_iterations, init_temperature, cooling_schedule)

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
    def run(self) -> Iterator[Optional[str]]:
        max_iterations = self.__config.max_iterations
        temperature = self.__config.init_temperature
        cooling_schedule = self.__config.cooling_schedule

        round_iter = range(max_iterations) if max_iterations is not None else iter(int, 1)
        for _ in round_iter:
            current_score = self.__problem.building.score

            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield
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
                yield

            temperature *= cooling_schedule
