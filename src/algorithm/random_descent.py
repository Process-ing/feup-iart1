from typing import override

from src.algorithm.algorithm import Algorithm
from src.model import RouterProblem


class RandomDescent(Algorithm):
    """
    Random Descent Algorithm
    Picks a random neighbor to explore, if it improves the score
    """
    def __init__(self, problem: RouterProblem) -> None:
        self.__problem = problem

    @override
    def run(self):
        while True:
            found_neighbor = False
            current_score = self.__problem.get_score(self.__problem.building)

            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield
                    continue

                if self.__problem.get_score(neighbor) > current_score:
                    self.__problem.building = neighbor
                    found_neighbor = True
                    break

            yield
            if not found_neighbor:
                break
