from algorithm.algorithm import Algorithm


class TabuSearch(Algorithm):
    """
    Tabu Search Algorithm
    Picks the best neighbor to explore (despite the score)
    """
    def __init__(self, problem: RouterProblem) -> None:
        self.__problem = problem
        self.__done = False

    @override
    def step(self):
        if self.__done:
            return

        best_neighbor = None
        best_score = self.__problem.get_score(self.__problem.building)
        for neighbor in self.__problem.building.get_neighborhood():
            if self.__problem.get_score(neighbor) > best_score:
                best_neighbor = neighbor
                best_score = self.__problem.get_score(neighbor)
        if best_neighbor is not None:
            self.__problem.building = best_neighbor
        else:
            self.__done = True

    @override
    def done(self) -> bool:
        return self.__done