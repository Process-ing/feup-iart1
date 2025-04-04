from typing import Tuple, cast, override
import numpy as np
from src.model.generic_building import GenericBuilding
from src.model.generic_problem import GenericRouterProblem
from src.model.building import Building

type BudgetInfo = Tuple[int, int, int]

class RouterProblem(GenericRouterProblem):
    def __init__(self, building: Building, router_range: int, budget_info: BudgetInfo) -> None:
        self.__building = building
        self.__router_range = router_range
        self.__backbone_price = budget_info[0]
        self.__router_price = budget_info[1]
        self.__budget = budget_info[2]

    @property
    @override
    def router_range(self) -> int:
        return self.__router_range

    @property
    def backbone_price(self) -> int:
        return self.__backbone_price

    @property
    def router_price(self) -> int:
        return self.__router_price

    @property
    def budget(self) -> int:
        return self.__budget

    @classmethod
    def from_text(cls, text: str) -> 'RouterProblem':
        parts = text.strip().split(maxsplit=8)

        initial_section = tuple(map(int, parts[0:8]))
        building_section = parts[8]

        rows, columns, router_range = initial_section[0:3]
        budget_info = cast(BudgetInfo, tuple(initial_section[3:6]))
        backbone = cast(Tuple[int, int], tuple(initial_section[6:8]))

        building = Building.from_text((rows, columns), backbone, building_section, None)
        problem = cls(building, router_range, budget_info)
        building.problem = problem

        return problem

    @property
    def building(self) -> Building:
        return self.__building

    @building.setter
    def building(self, building: Building) -> None:
        self.__building = building

    @override
    def check_budget(self, building: GenericBuilding) -> bool:
        num_routers = building.get_num_routers()
        num_connected_cells = building.get_num_connected_cells()

        return num_routers * self.__router_price + num_connected_cells \
            * self.backbone_price <= self.budget

    @override
    def get_score(self, building: GenericBuilding) -> int:
        num_routers = building.get_num_routers()
        num_connected_cells = building.get_num_connected_cells()
        coverage = building.get_coverage()

        return 1000 * coverage + \
            (self.budget - \
                (num_routers * self.router_price) - \
                (num_connected_cells * self.backbone_price))


    def dump_to_file(self, filename: str) -> None:
        building_map = self.__building.as_nparray()
        backbone_cells = [(i, j) for (i, j), cell in np.ndenumerate(building_map) \
            if cell & Building.BACKBONE_BIT]
        router_cells = [(i, j) for (i, j), cell in np.ndenumerate(building_map) \
            if cell & Building.ROUTER_BIT]

        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f'{len(backbone_cells)}\n')
            for (row, col) in backbone_cells:
                file.write(f'{row} {col}\n')
            file.write(f'{len(router_cells)}\n')
            for (row, col) in router_cells:
                file.write(f'{row} {col}\n')
