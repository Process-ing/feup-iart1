from dataclasses import dataclass
from typing import Dict, Iterator, Optional, override

from src.algorithm.algorithm import Algorithm, AlgorithmConfig
from src.model import RouterProblem

@dataclass
class RandomDescentConfig(AlgorithmConfig):
    max_neighborhood: Optional[int]
    max_iterations: Optional[int]

    @classmethod
    def from_flags(cls, flags: Dict[str, str]) -> Optional['RandomDescentConfig']:
        try:
            max_neighborhood = int(flags['max-neighborhood']) \
                if 'max-neighborhood' in flags else 5
            max_iterations = int(flags['max-iterations']) if 'max-iterations' in flags else None
        except ValueError:
            return None

        return cls(max_neighborhood=max_neighborhood, max_iterations=max_iterations)

class RandomDescent(Algorithm):
    '''
    Random Descent Algorithm
    Picks a random neighbor to explore, if it improves the score
    '''
    def __init__(self, problem: RouterProblem, config: RandomDescentConfig) -> None:
        self.__problem = problem
        self.__config = config

    @override
    def run(self) -> Iterator[Optional[str]]:
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
