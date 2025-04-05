from dataclasses import dataclass
from typing import Dict, Iterator, Optional, override

from src.algorithm.algorithm import Algorithm, AlgorithmConfig
from src.model import RouterProblem

@dataclass
class RandomDescentConfig(AlgorithmConfig):
    """
    Configuration class for the Random Descent algorithm.

    Attributes:
        max_neighborhood (Optional[int]): The maximum number of neighbors
        to explore in each iteration.
        max_iterations (Optional[int]): The maximum number of iterations
        to run the algorithm.
    """
    max_neighborhood: Optional[int]
    max_iterations: Optional[int]

    @classmethod
    def from_flags(cls, flags: Dict[str, str]) -> Optional['RandomDescentConfig']:
        """
        Creates a RandomDescentConfig instance from a dictionary of flags.

        Args:
            flags (Dict[str, str]): A dictionary where keys are configuration
            option names and values are their string representations.

        Returns:
            Optional[RandomDescentConfig]: An instance of RandomDescentConfig
            or None if flags are invalid.
        """
        if any(key not in ['max-neighborhood', 'max-iterations'] for key in flags):
            return None

        try:
            max_neighborhood = int(flags['max-neighborhood']) \
                if 'max-neighborhood' in flags else 5
            max_iterations = int(flags['max-iterations']) if 'max-iterations' in flags else None
        except ValueError:
            return None

        return cls(max_neighborhood=max_neighborhood, max_iterations=max_iterations)

class RandomDescent(Algorithm):
    """
    Random Descent algorithm for solving router placement problems.

    This algorithm explores random neighbors in the problem space and
    accepts the neighbor if it improves the score.
    The process continues until no further improvement is found or the
    maximum iterations are reached.
    """

    def __init__(self, problem: RouterProblem, config: RandomDescentConfig) -> None:
        """
        Initializes the RandomDescent instance with the given problem
        and configuration.

        Args:
            problem (RouterProblem): The problem instance containing the
            current building and constraints.
            config (RandomDescentConfig): The configuration parameters
            for the random descent algorithm.
        """
        self.__problem = problem
        self.__config = config

    @override
    def run(self) -> Iterator[Optional[str]]:
        """
        Executes the Random Descent algorithm by exploring the neighborhood
        for potential improvements.

        Yields:
            Optional[str]: A message indicating the placement or removal
            of a router, or `None` if no improvement is made.
        """
        max_neighborhood = self.__config.max_neighborhood
        max_iterations = self.__config.max_iterations

        round_iter = range(max_iterations) if max_iterations is not None else iter(int, 1)

        for _ in round_iter:
            best_neighbor_score = -1
            best_neighbor = None
            current_score = self.__problem.building.score

            neighbor_count = 0
            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield None
                    continue

                if neighbor.score > current_score:
                    if neighbor.score > best_neighbor_score:
                        best_neighbor_score = neighbor.score
                        best_neighbor = neighbor
                        yield f"{'Placed' if operator.place else 'Removed'} router at " \
                            f"({operator.row}, {operator.col})"

                    neighbor_count += 1
                    if neighbor_count == max_neighborhood:
                        break

                yield None

            if best_neighbor is not None:
                self.__problem.building = best_neighbor
            else:
                break
