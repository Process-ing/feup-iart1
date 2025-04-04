from dataclasses import dataclass
from typing import Dict, Iterator, Optional, override
from src.algorithm.algorithm import Algorithm
from src.model import RouterProblem

@dataclass
class RandomWalkConfig:
    max_iterations: Optional[int]

    @classmethod
    def from_flags(cls, flags: Dict[str, str]) -> Optional['RandomWalkConfig']:
        try:
            max_iterations = int(flags['max-iterations']) if 'max-iterations' in flags else None
        except ValueError:
            return None

        return cls(max_iterations)

class RandomWalk(Algorithm):
    '''
    Random Walk Algorithm
    Picks a random neighbor to explore (despite the score)
    '''
    def __init__(self, problem: RouterProblem, config: RandomWalkConfig) -> None:
        self.__problem = problem
        self.__config = config

    @override
    def run(self) -> Iterator[str]:
        max_iterations = self.__config.max_iterations

        round_iter = range(max_iterations) if max_iterations else iter(int, 1)

        for _ in round_iter:
            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield 'No neighbor found'
                    continue
                self.__problem.building = neighbor
                yield f"{'Placed' if operator.place else 'Removed'} router at " \
                f"({operator.row}, {operator.col})"
                break
