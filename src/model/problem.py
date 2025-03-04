from typing import cast
from .building import Building

type BudgetInfo = tuple[int, int, int]

class RouterProblem:
    def __init__(self, building: Building, router_range: int, budget_info: BudgetInfo):
        self.__building = building
        self.router_range = router_range
        self.backbone_price = budget_info[0]
        self.router_price = budget_info[1]
        self.budget = budget_info[2]

    @classmethod
    def from_text(cls, text: str) -> 'RouterProblem':
        parts = text.strip().split(maxsplit=8)

        initial_section = tuple(map(int, parts[0:8]))
        building_section = parts[8]

        rows, columns, router_range = initial_section[0:3]
        budget_info = cast(BudgetInfo, tuple(initial_section[3:6]))
        backbone = tuple(initial_section[6:8])

        building = Building.from_text(rows, columns, backbone, building_section)

        return cls(building, router_range, budget_info)

    @property
    def building(self) -> Building:
        return self.__building
