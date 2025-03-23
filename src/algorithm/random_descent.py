from typing import override

from src.algorithm.algorithm import Algorithm
from src.model import RouterProblem


class RandomDescent(Algorithm):
    """
    Random Descent Algorithm
    Picks a random neighbor to explore (despite the score)
    """
    def __init__(self, problem: RouterProblem) -> None:
        self.__problem = problem
        self.__done = False

    @override
    def step(self):
        if self.__done:
            return

        for neighbor in self.__problem.building.get_neighborhood():
            if self.__problem.get_score(neighbor) > self.__problem.get_score(self.__problem.building):
                self.__problem.building = neighbor
                return
        self.__done = True

    @override
    def done(self) -> bool:
        return self.__done