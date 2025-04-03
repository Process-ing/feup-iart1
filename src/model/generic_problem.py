from abc import abstractmethod
from src.model.generic_building import GenericBuilding

class GenericRouterProblem:
    @property
    @abstractmethod
    def router_range(self) -> int:
        pass

    @abstractmethod
    def check_budget(self, building: GenericBuilding) -> bool:
        pass

    @abstractmethod
    def get_score(self, building: GenericBuilding) -> int:
        pass
