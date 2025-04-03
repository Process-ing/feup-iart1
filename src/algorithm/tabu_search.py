import math
from collections import deque
from typing import Iterator, override
from src.algorithm.algorithm import Algorithm

class TabuSearch(Algorithm):
    """
    Tabu Search Algorithm
    Keeps track of visited solutions to avoid cycles and local optima.
    """
    def __init__(self, problem, tabu_tenure: int | None = None, neighborhood_len: int = 10,
                 max_iterations: int | None = None) -> None:
        if tabu_tenure is None:
            tabu_tenure = int(math.sqrt(problem.building.get_num_targets()) / problem.router_range)

        self.__problem = problem
        self.__tabu = deque(maxlen=tabu_tenure)
        self.__best_solution = problem.building
        self.__best_score = problem.get_score(problem.building)
        self.__neighborhood_len = neighborhood_len
        self.__max_iterations = max_iterations

    @override
    def run(self) -> Iterator[None]:
        for _ in range(self.__max_iterations) if self.__max_iterations is not None else iter(int, 1):
            best_pos = None
            best_neighbor = None
            best_score = float("-inf")
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

                score = self.__problem.get_score(neighbor)
                if score > best_score:
                    best_pos = (operator.row, operator.col)
                    best_neighbor = neighbor
                    best_score = score
                    best_operator = operator

                neighbor_count += 1
                if neighbor_count >= self.__neighborhood_len:
                    break

            if best_neighbor is None:  # Tabu tenure too long, whole neighborhood is tabu
                self.__tabu.popleft()
                yield "Tabu tenure too long"
                continue

            self.__problem.building = best_neighbor
            self.__tabu.append(best_pos)

            if best_score > self.__best_score:
                self.__best_solution = best_neighbor
                self.__best_score = best_score

            elif self.__tabu:
                self.__tabu.popleft()

            yield f"{"Placed" if best_operator.place else "Removed"} router at ({best_operator.row}, {best_operator.col})"

    def done(self) -> bool:
        return self.__done
