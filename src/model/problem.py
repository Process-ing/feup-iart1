from typing import cast
from .building import Building, CellType

type BudgetInfo = tuple[int, int, int]

class RouterProblem:
    def __init__(self, building: Building, router_range: int, budget_info: BudgetInfo, start_backbone : tuple[int, ...]):
        self.__building = building
        self.router_range = router_range
        self.backbone_price = budget_info[0]
        self.router_price = budget_info[1]
        self.budget = budget_info[2]
        self.start_backbone = start_backbone

    @classmethod
    def from_text(cls, text: str) -> 'RouterProblem':
        parts = text.strip().split(maxsplit=8)

        initial_section = tuple(map(int, parts[0:8]))
        building_section = parts[8]

        rows, columns, router_range = initial_section[0:3]
        budget_info = cast(BudgetInfo, tuple(initial_section[3:6]))
        backbone = tuple(initial_section[6:8])

        building = Building.from_text((rows, columns), backbone, building_section, router_range)

        return cls(building, router_range, budget_info, backbone)

    @property
    def building(self) -> Building:
        return self.__building

    def get_score(self) -> int:
        cost = 0
        covered = set()

        routers = self.__building.get_connected_routers(self.start_backbone)        

        # TODO(racoelhosilva): check optimizations
        for a, b in routers:
            cost += self.router_price
            for dx in range(-self.router_range, self.router_range + 1):
                    for dy in range(-self.router_range, self.router_range + 1):
                        x, y = a + dx, b + dy
                        if 0 <= x < self.__building.rows and 0 <= y < self.__building.columns and \
                            not self.__is_blocked((a,b), (x,y)):
                            covered.add((x,y))
        
        for a, b, cell in self.__building.iter():
            if (a,b) != self.start_backbone and cell & Building.BACKBONE_BIT:
                cost += self.backbone_price

        return 1000 * len(covered) + (self.budget - cost)

    def __is_blocked(self, router, cell):
        for w in range(min(router[0], cell[0]), max(router[0], cell[0]) + 1):
            for v in range(min(router[1], cell[1]), max(router[1], cell[1]) + 1):
                if self.__building.as_nparray()[w, v] == CellType.WALL.value:
                    return True
        return False

    def check_budget(self) -> int:
        cost = 0
        # TODO(racoelhosilva): check optimizations and cell types
        for _row, _col, cell in self.__building.iter():
            if cell == CellType.ROUTER:
                cost += self.router_price
            if cell & Building.BACKBONE_BIT:
                cost += self.backbone_price
        return cost <= self.budget
