from dataclasses import dataclass
from typing import Dict, Iterator, Optional, override
from src.algorithm.algorithm import Algorithm, AlgorithmConfig
from src.model import RouterProblem

@dataclass
class RandomWalkConfig(AlgorithmConfig):
    """
    Configuration class for the Random Walk algorithm.

    Attributes:
        max_iterations (Optional[int]): The maximum number of iterations to run the algorithm.
    """
    max_iterations: Optional[int]

    @classmethod
    def from_flags(cls, flags: Dict[str, str]) -> Optional['RandomWalkConfig']:
        """
        Creates a RandomWalkConfig instance from a dictionary of flags.

        Args:
            flags (Dict[str, str]): A dictionary where keys are configuration
            option names and values are their string representations.

        Returns:
            Optional[RandomWalkConfig]: An instance of RandomWalkConfig or
            None if flags are invalid.
        """
        if any(key not in ['max-iterations'] for key in flags):
            return None

        try:
            max_iterations = int(flags['max-iterations']) if 'max-iterations' in flags else None
        except ValueError:
            return None

        return cls(max_iterations)

class RandomWalk(Algorithm):
    """
    Random Walk algorithm for solving router placement problems.

    This algorithm explores random neighbors in the problem space, regardless
    of whether the neighbor improves the score.
    The process continues for a set number of iterations or until the algorithm is terminated.
    """
    def __init__(self, problem: RouterProblem, config: RandomWalkConfig) -> None:
        """
        Initializes the RandomWalk instance with the given problem and configuration.

        Args:
            problem (RouterProblem): The problem instance containing the
            current building and constraints.
            config (RandomWalkConfig): The configuration parameters for
            the random walk algorithm.
        """
        self.__problem = problem
        self.__config = config

    @override
    def run(self) -> Iterator[Optional[str]]:
        """
        Executes the Random Walk algorithm by randomly selecting a neighbor
        and making the move.

        Yields:
            Optional[str]: A message indicating the placement or removal of
            a router, or `None` if no valid neighbor is found.
        """
        max_iterations = self.__config.max_iterations

        round_iter = range(max_iterations) if max_iterations else iter(int, 1)

        for _ in round_iter:
            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield None
                    continue

                self.__problem.building = neighbor
                yield f"{'Placed' if operator.place else 'Removed'} router at " \
                f"({operator.row}, {operator.col})"
                break
