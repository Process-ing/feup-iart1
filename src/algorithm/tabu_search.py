import math
from collections import deque
from src.algorithm.algorithm import Algorithm

class TabuSearch(Algorithm):
    """
    Tabu Search Algorithm
    Keeps track of visited solutions to avoid cycles and local optima.
    """
    def __init__(self, problem, tabu_tenure: int | None = None, max_iterations: int = 10) -> None:
        if tabu_tenure is None:
            tabu_tenure = int(math.sqrt(problem.building.get_num_targets()) / problem.router_range)

        self.__problem = problem
        self.__tabu = deque(maxlen=tabu_tenure)
        self.__best_solution = problem.building
        self.__best_score = problem.get_score(problem.building)
        self.__max_iterations = max_iterations
        self.__done = False

    def step(self):
        if self.__done:
            return

        best_pos = None
        best_neighbor = None
        best_score = float("-inf")
        iterations = 0

        for operator in self.__problem.building.get_neighborhood():
            if (operator.row, operator.col) in self.__tabu:
                continue

            neighbor = operator.apply(self.__problem.building)
            if not neighbor:
                continue

            score = self.__problem.get_score(neighbor)
            if score > best_score:
                best_pos = (operator.row, operator.col)
                best_neighbor = neighbor
                best_score = score

            iterations += 1
            if iterations >= self.__max_iterations:
                break

        self.__problem.building = best_neighbor

        if best_score > self.__best_score:
            self.__best_solution = best_neighbor
            self.__best_score = best_score
            self.__tabu.append(best_pos)
        elif self.__tabu:
            self.__tabu.popleft()

    def done(self) -> bool:
        return self.__done
