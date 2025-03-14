from typing import cast
from .building import Building, CellType

type BudgetInfo = tuple[int, int, int]

class RouterProblem:
    def __init__(self, building: Building, router_range: int, \
                 budget_info: BudgetInfo, start_backbone : tuple[int, int]):
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
        backbone = cast(tuple[int, int], tuple(initial_section[6:8]))

        building = Building.from_text((rows, columns), backbone, building_section, router_range)

        return cls(building, router_range, budget_info, backbone)

    @property
    def building(self) -> Building:
        return self.__building

    def get_score(self) -> int:
        cost = 0
        covered = set()

        routers, backbones = self.__building.get_connected_routers(self.start_backbone)

        # TODO(racoelhosilva): check optimizations
        for a, b in routers:
            cost += self.router_price
            for dx in range(-self.router_range, self.router_range + 1):
                for dy in range(-self.router_range, self.router_range + 1):
                    x, y = a + dx, b + dy
                    if 0 <= x < self.__building.rows and 0 <= y < self.__building.columns and \
                        not self.__is_blocked((a,b), (x,y)):
                        covered.add((x,y))
        cost += (len(backbones) - 1) * self.backbone_price

        return 1000 * len(covered) + (self.budget - cost)

    def __is_blocked(self, router: tuple[int, int], cell: tuple[int, int]) -> bool:
        for w in range(min(router[0], cell[0]), max(router[0], cell[0]) + 1):
            for v in range(min(router[1], cell[1]), max(router[1], cell[1]) + 1):
                if self.__building.as_nparray()[w, v] == CellType.WALL.value:
                    return True
        return False

    def check_budget(self) -> int:
        # TODO(racoelhosilva): check optimizations and cell types
        routers, backbones = self.__building.get_connected_routers(self.start_backbone)
        cost = len(routers) * self.router_price + (len(backbones) - 1) * self.backbone_price
        return cost <= self.budget

    def dump_to_file(self, filename: str) -> None:
        reverse_char_cell_map = {v.value: k for k, v in Building.CHAR_CELL_MAP.items()}
        reverse_char_cell_map[CellType.ROUTER.value] = ord('R')
        reverse_char_cell_map[CellType.WALL.value] = ord('#')
        reverse_char_cell_map[CellType.TARGET.value] = ord('.')
        reverse_char_cell_map[CellType.VOID.value] = ord('-')

        with open(filename, 'w', encoding='utf-8') as file:
            for row in self.__building.as_nparray():
                file.write(''.join(
                    'B' if cell & Building.BACKBONE_BIT else \
                        chr(reverse_char_cell_map[cell & Building.CELL_TYPE_MASK])
                    for cell in row
                ) + '\n')
