from dataclasses import dataclass
from collections import deque
import math
from typing import Deque, Iterator, Tuple, Union, override
from src.model.building import Building
from src.model.problem import RouterProblem
from src.algorithm.algorithm import Algorithm

type TabuList = Deque[Tuple[int, int]]

@dataclass
class TabuSearchConfig:
    tabu_tenure: int
    max_iterations: Union[int, None]
    max_neighborhood: Union[int, None]

class TabuSearch(Algorithm):
    '''
    Tabu Search Algorithm
    Keeps track of visited solutions to avoid cycles and local optima.
    '''
    def __init__(self, problem: RouterProblem, config: TabuSearchConfig) -> None:
        self.__problem = problem
        self.__best_solution = problem.building
        self.__best_score = problem.building.score
        self.__config = config

    @override
    def run(self) -> Iterator[str]:
        max_iterations = self.__config.max_iterations
        max_neighborhood = self.__config.max_neighborhood
        tabu_tenure = self.__config.tabu_tenure

        tabu_list: TabuList = deque(maxlen=tabu_tenure)

        round_iter = range(max_iterations) if max_iterations is not None else iter(int, 1)
        for _ in round_iter:
            best_pos = None
            best_neighbor = None
            best_score = -1
            neighbor_count = 0
            best_operator = None

            for operator in self.__problem.building.get_neighborhood():
                if (operator.row, operator.col) in tabu_list:
                    yield 'Neighbor is tabu'
                    continue

                neighbor = operator.apply(self.__problem.building)
                if not neighbor:
                    yield 'No neighbor found'
                    continue

                if neighbor.score > best_score:
                    best_neighbor = neighbor
                    best_score = neighbor.score
                    best_operator = operator

                neighbor_count += 1
                if neighbor_count >= max_neighborhood:
                    break

            if best_neighbor is None or best_pos is None:
                # Tabu tenure too long, whole neighborhood is tabu
                tabu_list.popleft()
                yield 'Tabu tenure too long'
                continue

            self.__problem.building = best_neighbor
            tabu_list.append(best_pos)

            if best_score and best_score > self.__best_score:
                self.__best_solution = best_neighbor
                self.__best_score = best_score

            elif tabu_list:
                tabu_list.popleft()

            if best_operator is None:
                yield 'No operator found'
                continue

            action = 'Placed' if best_operator.place else 'Removed'
            row, col = best_operator.row, best_operator.col
            yield f'{action} router at ({row}, {col})'

    @property
    def best_solution(self) -> Building:
        return self.__best_solution

    @staticmethod
    def get_default_tenure(problem: RouterProblem) -> int:
        targets = problem.building.get_num_targets()
        router_range = problem.router_range

        return int(math.sqrt(targets) / router_range * 2)
