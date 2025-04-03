import math
from collections import deque
from typing import Deque, Iterator, Tuple, override
from src.model.building import Building
from src.model.problem import RouterProblem
from src.algorithm.algorithm import Algorithm

type TabuTable = Deque[Tuple[int, int]]

class TabuSearch(Algorithm):
    """
    Tabu Search Algorithm
    Keeps track of visited solutions to avoid cycles and local optima.
    """
    def __init__(self, problem: RouterProblem, tabu_tenure: int | None = None,
                 neighborhood_len: int = 10, max_iterations: int | None = None) -> None:
        if tabu_tenure is None:
            tabu_tenure = int(math.sqrt(problem.building.get_num_targets()) / problem.router_range)

        self.__problem = problem
        self.__tabu: TabuTable = deque(maxlen=tabu_tenure)
        self.__best_solution = problem.building
        self.__best_score = problem.building.score
        self.__neighborhood_len = neighborhood_len
        self.__max_iterations = max_iterations

    @override
    def run(self) -> Iterator[None]:
        round_iter = range(self.__max_iterations) \
            if self.__max_iterations is not None else iter(int, 1)
        for _ in round_iter:
            best_pos = None
            best_neighbor = None
            best_score = -1
            neighbor_count = 0
            best_operator = None

            for operator in self.__problem.building.get_neighborhood():
                if (operator.row, operator.col) in self.__tabu:
                    yield "Neighbor is tabu"
                    continue

                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield "No neighbor found"
                    continue

                if neighbor.score > best_score:
                    best_pos = (operator.row, operator.col)
                    best_neighbor = neighbor
                    best_score = neighbor.score
                    best_operator = operator

                neighbor_count += 1
                if neighbor_count >= self.__neighborhood_len:
                    break

            if best_neighbor is None or best_pos is None:
                # Tabu tenure too long, whole neighborhood is tabu
                self.__tabu.popleft()
                yield "Tabu tenure too long"
                continue

            self.__problem.building = best_neighbor
            self.__tabu.append(best_pos)

            if best_score and best_score > self.__best_score:
                self.__best_solution = best_neighbor
                self.__best_score = best_score

            elif self.__tabu:
                self.__tabu.popleft()

            yield f"{'Placed' if best_operator.place else 'Removed'} router at ({best_operator.row}, {best_operator.col})"
    def done(self) -> bool:
        return self.__done
        
    @property
    def best_solution(self) -> Building:
        return self.__best_solution
