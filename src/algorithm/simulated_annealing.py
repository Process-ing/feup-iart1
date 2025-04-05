from dataclasses import dataclass
from typing import Iterator, Optional, override
import math
import random

from src.model.problem import RouterProblem
from src.algorithm.algorithm import Algorithm, AlgorithmConfig

@dataclass
class SimulatedAnnealingConfig(AlgorithmConfig):
    """
    Configuration class for the Simulated Annealing algorithm.

    Attributes:
        max_iterations (Optional[int]): The maximum number of iterations
        to run the algorithm.
        init_temperature (float): The initial temperature for the annealing process.
        cooling_schedule (float): The rate at which the temperature decreases during
        the algorithm's execution.
    """
    max_iterations: Optional[int]
    init_temperature: float
    cooling_schedule: float

    @classmethod
    def from_flags(cls, flags: dict[str, str]) -> Optional['SimulatedAnnealingConfig']:
        """
        Creates a SimulatedAnnealingConfig instance from a dictionary of flags.

        Args:
            flags (dict[str, str]): A dictionary where keys are configuration
            option names and values are their string representations.

        Returns:
            Optional[SimulatedAnnealingConfig]: An instance of SimulatedAnnealingConfig
            or None if flags are invalid.
        """
        if any(key not in ['max-iterations', 'init-temperature',
                            'cooling-schedule'] for key in flags):
            return None

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
    """
    Simulated Annealing algorithm for solving router placement problems.

    This algorithm explores random neighbors and accepts them based on their score.
    If the neighbor's score is better, it is chosen. 
    Otherwise, the algorithm accepts the neighbor with a certain probability,
    which depends on the temperature and a cooling schedule.
    The temperature decreases according to the cooling schedule.
    """
    def __init__(self, problem: RouterProblem, config: SimulatedAnnealingConfig) -> None:
        """
        Initializes the SimulatedAnnealing instance with the given problem and configuration.

        Args:
            problem (RouterProblem): The problem instance containing the
            current building and constraints.
            config (SimulatedAnnealingConfig): The configuration parameters
            for the simulated annealing algorithm.
        """
        self.__problem = problem
        self.__config = config

    @override
    def run(self) -> Iterator[Optional[str]]:
        """
        Executes the Simulated Annealing algorithm by randomly selecting a neighbor
        and deciding whether to accept it.

        The algorithm accepts a neighbor if its score is better or based on a
        probability determined by the temperature and cooling schedule.

        Yields:
            Optional[str]: A message indicating the placement or removal of a router,
            or `None` if no valid neighbor is found.
        """
        max_iterations = self.__config.max_iterations
        temperature = self.__config.init_temperature
        cooling_schedule = self.__config.cooling_schedule

        round_iter = range(max_iterations) if max_iterations is not None else iter(int, 1)
        for _ in round_iter:
            current_score = self.__problem.building.score

            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield None
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
                yield None

            temperature *= cooling_schedule
