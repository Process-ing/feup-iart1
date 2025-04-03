from typing import Iterator, override
from src.algorithm.algorithm import Algorithm
from src.model import RouterProblem

class RandomWalk(Algorithm):
    '''
    Random Walk Algorithm
    Picks a random neighbor to explore (despite the score)
    '''
    def __init__(self, problem: RouterProblem, max_iterations: int | None = None) -> None:
        self.__problem = problem
        self.__max_iterations = max_iterations

    @override
    def run(self) -> Iterator[str]:
        for _ in range(self.__max_iterations) if self.__max_iterations else iter(int, 1):
            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield 'No neighbor found'
                    continue
                self.__problem.building = neighbor
                yield f"{'Placed' if operator.place else 'Removed'} router at " \
                f"({operator.row}, {operator.col})"
                break
