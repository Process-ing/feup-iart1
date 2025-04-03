from typing import override

from src.algorithm.algorithm import Algorithm
from src.model import RouterProblem


class RandomDescent(Algorithm):
    """
    Random Descent Algorithm
    Picks a random neighbor to explore, if it improves the score
    """
    def __init__(self, problem: RouterProblem, max_iterations: int = 2000000000000000) -> None:
        self.__problem = problem
        self.__max_iterations = max_iterations

    @override
    def run(self):
        for _ in range(self.__max_iterations):
            found_neighbor = False
            current_score = self.__problem.get_score(self.__problem.building)

            for operator in self.__problem.building.get_neighborhood():
                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield "No neighbor found"
                    continue

                if self.__problem.get_score(neighbor) > current_score:
                    self.__problem.building = neighbor
                    found_neighbor = True
                    yield f"{'Placed' if operator.place else 'Removed'} router at ({operator.row}, {operator.col})"
                    break   
                yield "Neighbor not improving score"

            if not found_neighbor:
                break
