from typing import override
from src.algorithm import Algorithm
from src.model import RouterProblem

class RandomWalk(Algorithm):
    """
    Random Walk Algorithm
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
            self.__problem.building = neighbor
            return
        self.__done = True

    @override
    def done(self) -> bool:
        return self.__done
