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
        self.__done = False

    @override
    def step(self):
        if self.__done:
            return

        current_score = self.__problem.get_score(self.__problem.building)

        for operator in self.__problem.building.get_neighborhood():
            neighbor = operator.apply(self.__problem.building)
            if not neighbor:
                continue
            
            if self.__problem.get_score(neighbor) > current_score:
                self.__problem.building = neighbor
                return
        self.__done = True

    @override
    def done(self) -> bool:
        return self.__done