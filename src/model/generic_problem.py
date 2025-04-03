from abc import abstractmethod
from typing import TypeVar

from src.model.generic_building import GenericBuilding

class GenericRouterProblem:
    @abstractmethod
    def check_budget(self, building: GenericBuilding) -> bool:
        pass

    @abstractmethod
    def get_score(self, building: GenericBuilding) -> int:
        pass