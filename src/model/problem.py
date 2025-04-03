from typing import Callable, cast
import numpy as np
from src.model.building import Building, CellType, CheckBudgetCallback

type BudgetInfo = tuple[int, int, int]

class RouterProblem:
    def __init__(self, building: Building, router_range: int, budget_info: BudgetInfo,
                 start_backbone : tuple[int, int], check_budget: CheckBudgetCallback) -> None:
        self.__building = building
        self.router_range = router_range
        self.backbone_price = budget_info[0]
        self.router_price = budget_info[1]
        self.budget = budget_info[2]
        self.start_backbone = start_backbone
        self.check_budget = check_budget

    @classmethod
    def from_text(cls, text: str) -> 'RouterProblem':
        parts = text.strip().split(maxsplit=8)

        initial_section = tuple(map(int, parts[0:8]))
        building_section = parts[8]

        rows, columns, router_range = initial_section[0:3]
        budget_info = cast(BudgetInfo, tuple(initial_section[3:6]))
        backbone = cast(tuple[int, int], tuple(initial_section[6:8]))
        check_budget = cls.__gen_check_budget(budget_info[1], budget_info[0], budget_info[2])

        building = Building.from_text((rows, columns), backbone, building_section, router_range, check_budget)

        return cls(building, router_range, budget_info, backbone, check_budget)

    @property
    def building(self) -> Building:
        return self.__building

    @building.setter
    def building(self, building: Building) -> None:
        self.__building = building

    def get_score(self, building: Building) -> int:
        num_routers = building.get_num_routers()
        num_connected_cells = building.get_num_connected_cells()
        coverage = building.get_coverage()

        return 1000 * coverage + \
            (self.budget - \
                (num_routers * self.router_price) - \
                (num_connected_cells * self.backbone_price))

    @staticmethod
    def __gen_check_budget(router_price: int, backbone_price: int, budget: int) -> CheckBudgetCallback:
        def check_budget(building: Building) -> bool:
            num_routers = building.get_num_routers()
            num_connected_cells = building.get_num_connected_cells()

            return num_routers * router_price + num_connected_cells * backbone_price <= budget

        return check_budget

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


