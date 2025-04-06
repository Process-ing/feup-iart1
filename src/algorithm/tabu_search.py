from dataclasses import dataclass
from collections import deque
import math
from typing import Deque, Iterator, Optional, Tuple, override
from src.model import RouterProblem
from src.algorithm.algorithm import Algorithm, AlgorithmConfig

type TabuList = Deque[Tuple[int, int]]

@dataclass
class TabuSearchConfig(AlgorithmConfig):
    """
    Configuration class for the Tabu Search algorithm.

    Attributes:
        tabu_tenure (int): The maximum number of iterations a solution is
        forbidden (tabu) before it can be revisited.
        max_iterations (Optional[int]): The maximum number of iterations
        to run the algorithm.
        max_neighborhood (Optional[int]): The maximum number of neighbors
        to explore in each iteration.
    """
    tabu_tenure: int
    max_iterations: Optional[int]
    max_neighborhood: Optional[int]

    @classmethod
    def from_flags(cls, flags: dict[str, str],
                   default_tabu_tenure: int) -> Optional['TabuSearchConfig']:
        """
        Creates a TabuSearchConfig instance from a dictionary of flags.

        Args:
            flags (dict[str, str]): A dictionary where keys are configuration
            option names and values are their string representations.
            default_tabu_tenure (int): The default tabu tenure value.

        Returns:
            Optional[TabuSearchConfig]: An instance of TabuSearchConfig
            or None if flags are invalid.
        """
        if any(key not in ['tabu-tenure', 'max-iterations', 'max-neighborhood'] for key in flags):
            return None

        try:
            tabu_tenure = int(flags['tabu-tenure']) \
                if 'tabu-tenure' in flags else default_tabu_tenure
            max_iterations = int(flags['max-iterations']) if 'max-iterations' in flags else None
            max_neighborhood = int(flags['max-neighborhood']) \
                if 'max-neighborhood' in flags else 10
        except ValueError:
            return None

        return cls(tabu_tenure, max_iterations, max_neighborhood)

class TabuSearch(Algorithm):
    """
    Tabu Search algorithm for solving router placement problems.

    This algorithm explores neighbors of the current solution while keeping
    track of visited solutions to avoid cycles and local optima. 
    The algorithm uses a tabu list to forbid revisiting recent solutions for
    a specified number of iterations.

    Methods:
        is_tabu: Checks if a given solution is in the tabu list.
        run: Executes the Tabu Search algorithm by exploring neighbors and
        applying the best solution while respecting the tabu list.
        get_default_tenure: Calculates the default tabu tenure based on the
        problem's parameters.
    """
    def __init__(self, problem: RouterProblem, config: TabuSearchConfig) -> None:
        """
        Initializes the TabuSearch instance with the given problem and configuration.

        Args:
            problem (RouterProblem): The problem instance containing the
            current building and constraints.
            config (TabuSearchConfig): The configuration parameters for
            the tabu search algorithm.
        """
        self.__problem = problem
        self.__config = config

    def is_tabu(self, tabu_list: TabuList, pos: Tuple[int, int]) -> bool:
        """
        Checks if a given position is in the tabu list, indicating it
        is forbidden for exploration.

        Args:
            tabu_list (TabuList): The list of positions that are currently forbidden.
            pos (Tuple[int, int]): The position to check.

        Returns:
            bool: True if the position is tabu, False otherwise.
        """
        tabu_range = (self.__problem.router_range + 1) // 2
        row, col = pos

        return any(abs(row - trow) <= tabu_range and abs(col - tcol) <= tabu_range
                   for trow, tcol in tabu_list)

    @override
    def run(self) -> Iterator[Optional[str]]:
        """
        Executes the Tabu Search algorithm by selecting the best neighboring
        solution while considering the tabu list.

        The algorithm explores a neighborhood of solutions and keeps track
        of the best one that is not tabu. If no valid solution is found,
        the algorithm moves to the next iteration. The tabu list is updated
        with the current solution's position after each iteration.

        Yields:
            Optional[str]: A message indicating the placement or removal of
            a router, or a status message like "Tabu tenure too long."
        """
        max_iterations = self.__config.max_iterations
        max_neighborhood = self.__config.max_neighborhood
        tabu_tenure = self.__config.tabu_tenure

        tabu_list: TabuList = deque(maxlen=tabu_tenure)

        round_iter = range(max_iterations) if max_iterations is not None else iter(int, 1)
        for _ in round_iter:
            best_neighbor = None
            best_score = -1
            best_operator = None
            neighbor_count = 0

            for operator in self.__problem.building.get_neighborhood():
                if self.is_tabu(tabu_list, operator.pos):
                    continue

                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield None
                    continue

                if neighbor.score > best_score:
                    best_neighbor = neighbor
                    best_score = neighbor.score
                    best_operator = operator

                neighbor_count += 1
                if max_neighborhood is not None and neighbor_count >= max_neighborhood:
                    break

                yield None

            if best_neighbor is None or best_operator is None:
                # Tabu tenure too long, whole neighborhood is tabu
                tabu_list.popleft()
                yield 'Tabu tenure too long'
                continue

            self.__problem.building = best_neighbor
            tabu_list.append(best_operator.pos)

            if best_operator is None:
                yield 'No operator found'
                continue

            action = 'Placed' if best_operator.place else 'Removed'
            row, col = best_operator.row, best_operator.col
            yield f'{action} router at ({row}, {col})'

    @staticmethod
    def get_default_tenure(problem: RouterProblem) -> int:
        """
        Calculates the default tabu tenure based on the problem's parameters.

        Args:
            problem (RouterProblem): The problem instance containing
            the current building and constraints.

        Returns:
            int: The default tabu tenure value based on the problem's characteristics.
        """
        targets = problem.building.get_num_targets()
        router_range = problem.router_range

        return int(math.sqrt(targets) / router_range * 2)
