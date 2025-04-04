from enum import Enum
from typing import Callable, Deque, Dict, Iterator, Optional, Set, Tuple, cast, List, override
from collections import deque
from copy import deepcopy
import random
import numpy as np

from src.model.generic_building import GenericBuilding
from src.model.generic_problem import GenericRouterProblem
from src.model.disjoint_set import DisjointSet
from src.model.error import ProblemLoadError

type CellArray = np.ndarray[Tuple[int, ...], np.dtype[np.uint8]]
type CheckBudgetCallback = Callable[['Building'], bool]
type Pos = Tuple[int, int]
type SteinerTreeQueue = Deque[Tuple[Pos, Pos, Optional[Pos], Optional[Pos]]]

class Operator:
    def __init__(self, place: bool, row: int, col: int) -> None:
        self.place = place
        self.row = row
        self.col = col

    def apply(self, building: 'Building') -> Optional['Building']:
        new_building = building.copy()

        if self.place:
            success = new_building.place_router(self.row, self.col)
        else:
            success = new_building.remove_router(self.row, self.col)

        assert building.problem is not None
        return new_building if success and building.problem.check_budget(new_building) else None

    @property
    def pos(self) -> Pos:
        return (self.row, self.col)

class CellType(Enum):
    VOID = 0
    TARGET = 1
    WALL = 2

class Building(GenericBuilding):
    BACKBONE_BIT = np.uint8(1 << 7)  # Marks a cell connected to the backbone
    COVERED_BIT = np.uint8(1 << 6)  # Marks a cell covered by a router
    ROUTER_BIT = np.uint8(1 << 5)  # Marks a cell as a router
    CELL_TYPE_MASK = np.uint8(0b00011111)  # Mask for the cell type

    CHAR_CELL_MAP = {
        ord('-'): CellType.VOID,
        ord('.'): CellType.TARGET,
        ord('#'): CellType.WALL,
    }

    def __init__(self, cells: CellArray, backbone: Pos, new_router_probability: float,
                 problem: Optional[GenericRouterProblem]) -> None:
        self.__cells: CellArray = cells
        self.__backbone_root = backbone
        self.__new_router_probability = new_router_probability
        self.problem = problem
        self.__score: Optional[int] = None

    def copy(self) -> 'Building':
        return Building(deepcopy(self.__cells), self.__backbone_root,
                   self.__new_router_probability, self.problem)

    @classmethod
    def from_text(cls, shape: Tuple[int, int], backbone: Pos,
                  text: str, problem: Optional[GenericRouterProblem]) -> 'Building':
        rows, columns = shape
        if rows < 1 or columns < 1:
            raise ProblemLoadError(f'Invalid building size {rows}x{columns}')
        if backbone[0] < 0 or backbone[0] >= rows or backbone[1] < 0 or backbone[1] >= columns:
            raise ProblemLoadError(f'Invalid backbone position ({backbone[0], backbone[1]})')

        text = text.replace('\n', '')
        if len(text) != rows * columns:
            raise ProblemLoadError('Invalid text size')

        cells = np.frombuffer(text.encode(), dtype=np.uint8)
        cells = cells.reshape((rows, columns)).copy()

        for (row, col), cell_byte in np.ndenumerate(cells):
            cell = int(cell_byte)
            if cell in cls.CHAR_CELL_MAP:
                cells[row, col] = cls.CHAR_CELL_MAP[cell].value
            else:
                raise ProblemLoadError(f'Invalid character in text \'{chr(cell)}\'')

        cells[backbone] |= cls.BACKBONE_BIT

        return cls(cells, backbone, 0.8, problem)

    @property
    def rows(self) -> int:
        return self.__cells.shape[0]

    @property
    def columns(self) -> int:
        return self.__cells.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return cast(Tuple[int, int], self.__cells.shape)

    def get_routers(self) -> List[Tuple[int, int]]:
        return list(zip(*np.where(self.__cells & self.ROUTER_BIT)))

    def get_target_cells(self) -> List[Tuple[int, int]]:
        return list(zip(*np.where(self.__cells & self.CELL_TYPE_MASK == CellType.TARGET.value)))

    def __str__(self) -> str:
        return '\n'.join(''.join(map(chr, row)) for row in self.__cells)

    def as_nparray(self) -> CellArray:
        return self.__cells.copy()

    def as_nparray_transposed(self) -> CellArray:
        return self.__cells.transpose()

    def iter(self) -> Iterator[Tuple[int, int, int]]:
        return ((row, column, int(cell)) for (row, column), cell in np.ndenumerate(self.__cells))

    def get_connected_routers(self, root: Pos) \
        -> Tuple[Set[Pos], Set[Tuple[int, int]]]:
        routers = set()
        backbones = set()
        queue = deque([root])
        backbones.add(root)
        directions = [0, 1, 0, -1, 0]

        while queue:
            row, col = queue.popleft()
            if (self.__cells[row, col] & self.ROUTER_BIT) != 0:
                routers.add((row,col))
            for i in range(4):
                nr, nc = row + directions[i], col + directions[i+1]
                if 0 <= nr < self.rows and 0 <= nc < self.columns and \
                        (nr, nc) not in backbones and \
                        self.__cells[nr, nc] & self.BACKBONE_BIT:
                    queue.append((nr, nc))
                    backbones.add((nr, nc))
        return routers, backbones

    def cover_neighbors(self, row: int, col: int) -> None:
        assert self.problem is not None
        rrange = self.problem.router_range

        row_start = max(0, row - rrange)
        row_len = min(self.__cells.shape[0] - row_start, 2 * rrange + 1)
        col_start = max(0, col - rrange)
        col_len = min(self.__cells.shape[1] - col_start, 2 * rrange + 1)

        ctr_row = row - row_start
        ctr_col = col - col_start

        neighborhood = np.zeros((row_len, col_len), dtype=np.uint8)
        neighborhood |= self.__cells[row_start:row_start + row_len, \
            col_start:col_start + col_len] & Building.CELL_TYPE_MASK

        neighborhood[ctr_row, ctr_col] |= self.COVERED_BIT

        line_iters = [
            ((ctr_row, ncol) for ncol in range(ctr_col + 1, col_len)),
            ((ctr_row, ncol) for ncol in range(ctr_col - 1, -1, -1)),
            ((nrow, ctr_col) for nrow in range(ctr_row + 1, row_len)),
            ((nrow, ctr_col) for nrow in range(ctr_row - 1, -1, -1))
        ]

        for line_iter in line_iters:
            for nrow, ncol in line_iter:
                if neighborhood[nrow, ncol] & self.CELL_TYPE_MASK == CellType.WALL.value:
                    break
                neighborhood[nrow, ncol] |= self.COVERED_BIT

        square_iters = [
            (((nrow, ncol) for nrow in range(ctr_row - 1, -1, -1)
                for ncol in range(ctr_col - 1, -1, -1)), 1, 1),
            (((nrow, ncol) for nrow in range(ctr_row - 1, -1, -1)
                for ncol in range(ctr_col + 1, col_len)), 1, -1),
            (((nrow, ncol) for nrow in range(ctr_row + 1, row_len)
                for ncol in range(ctr_col - 1, -1, -1)), -1, 1),
            (((nrow, ncol) for nrow in range(ctr_row + 1, row_len)
                for ncol in range(ctr_col + 1, col_len)), -1, -1),
        ]

        for square_iter, rstep, cstep in square_iters:
            for nrow, ncol in square_iter:
                if neighborhood[nrow, ncol] & self.CELL_TYPE_MASK != CellType.WALL.value \
                    and neighborhood[nrow + rstep, ncol] & self.COVERED_BIT \
                    and neighborhood[nrow, ncol + cstep] & self.COVERED_BIT:
                    neighborhood[nrow, ncol] |= self.COVERED_BIT

        self.__cells[row_start:row_start + row_len, col_start:col_start + col_len] |= neighborhood

    def place_router(self, row: int, column: int) -> bool:

        current_cell = self.__cells[row, column]

        # Check if position is valid (routers cannot be placed inside walls)
        if current_cell & self.CELL_TYPE_MASK == CellType.WALL.value \
            or current_cell & self.CELL_TYPE_MASK == CellType.VOID.value:
            return False

        # Check if router is already placed
        if (current_cell & self.ROUTER_BIT) != 0:
            return False

        # Place the router
        self.__cells[row, column] |= self.ROUTER_BIT

        # Connect the router to the backbone (BFS)
        queue = deque([(row, column)])
        visited = np.zeros((self.rows, self.columns), dtype=bool)
        visited[(row, column)] = True
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            r, c = queue.popleft()

            if self.__cells[r, c] & self.BACKBONE_BIT:
                while (r, c) != (row, column):
                    self.__cells[r, c] |= self.BACKBONE_BIT
                    r, c = parent[(r, c)]
                self.__cells[r, c] |= self.BACKBONE_BIT
                break

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.columns and not visited[nr, nc]:
                    queue.append((nr, nc))
                    visited[(nr, nc)] = True
                    parent[(nr, nc)] = (r, c)


        # Update router coverage
        self.cover_neighbors(row, column)

        return True

    def update_neighbor_coverage(self, row: int, column: int) -> None:
        assert self.problem is not None
        router_range = self.problem.router_range

        cell_row_start = max(0, row - router_range)
        cell_row_end = min(self.__cells.shape[0], row + router_range + 1)
        cell_col_start = max(0, column - router_range)
        cell_col_end = min(self.__cells.shape[1], column + router_range + 1)

        self.__cells[cell_row_start:cell_row_end, cell_col_start:cell_col_end] &= ~self.COVERED_BIT

        router_row_start = max(0, row - 2 * router_range)
        router_row_end = min(self.__cells.shape[0], row + 2 * router_range + 1)
        router_col_start = max(0, column - 2 * router_range)
        router_col_end = min(self.__cells.shape[1], column + 2 * router_range + 1)

        for rrow in range(router_row_start, router_row_end):
            for rcol in range(router_col_start, router_col_end):
                if self.__cells[rrow, rcol] & self.ROUTER_BIT:
                    self.__cells[rrow, rcol] |= self.COVERED_BIT
                    self.cover_neighbors(rrow, rcol)

    def remove_router(self, row: int, column: int) -> bool:
        if (self.__cells[row, column] & self.ROUTER_BIT) == 0:
            return False

        max_row, max_col = self.rows, self.columns
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        queue: Deque[Tuple[int, int]] = deque()
        queue.append((row, column))

        while queue:
            r, c = queue.popleft()

            self.__cells[r, c] &= ~(self.ROUTER_BIT | self.BACKBONE_BIT) \
                if (r, c) != self.__backbone_root \
                else ~self.ROUTER_BIT

            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < max_row and 0 <= new_c < max_col and \
                    (self.__cells[new_r, new_c] & self.BACKBONE_BIT) != 0 and \
                    (self.__cells[new_r, new_c] & self.ROUTER_BIT) == 0 and \
                    (new_r, new_c) != self.__backbone_root:
                    queue.append((new_r, new_c))

        self.reconnect_routers()
        self.update_neighbor_coverage(row, column)
        return True

    def reconnect_routers(self) -> None:
        routers = list(zip(*np.where(self.__cells & self.BACKBONE_BIT)))
        if not routers:
            return

        def reconstruct_path(pred: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
                             p: Optional[Tuple[int, int]]) -> Set[Tuple[int, int]]:
            res = set()
            while p:
                res.add(p)
                p = pred.get(p, None)
            return res

        def steiner_tree(grid: CellArray, terminals: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
            rows, cols = len(grid), len(grid[0])
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

            terminal_indices = {t: i for i, t in enumerate(terminals)}
            dsu = DisjointSet(len(terminals))
            queue: SteinerTreeQueue = deque()
            source = {}
            pred = {}
            res = set(terminals)

            for (x, y) in terminals:
                queue.append(((x, y), (x, y), None, None))

            while queue and dsu.forests > 1:
                (x, y), src, p1, p2 = queue.popleft()

                if (x,y) not in source:
                    source[(x, y)] = src
                    pred[(x, y)] = p1

                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and (nx,ny) not in source:
                            queue.append(((nx, ny), src, (x, y), None))

                    continue

                if dsu.connected(terminal_indices[source[(x,y)]], terminal_indices[src]):
                    continue

                if (x, y) in terminals:
                    dsu.union(terminal_indices[source[(x, y)]], terminal_indices[src])
                    res |= reconstruct_path(pred, p1)
                    res |= reconstruct_path(pred, p2)
                else:
                    queue.append((source[(x, y)], src, p1, (x, y)))

            return res

        tree_cells = steiner_tree(self.__cells, [self.__backbone_root] + routers)

        for row, col in tree_cells:
            self.__cells[row, col] |= self.BACKBONE_BIT

    def get_neighborhood(self) -> Iterator[Operator]:
        """
        Generates neighboring building configurations by placing or removing routers.

        This method shuffles the list of routers and target cells, then iteratively
        creates new building configurations by either placing a router in a target cell
        or removing a router from its current position. The decision to place or remove
        a router is based on a predefined probability.

        Yields:
            Building: A new building configuration with a router placed or removed.
        """
        routers = self.get_routers()
        targets = self.get_target_cells()
        random.shuffle(routers)
        random.shuffle(targets)

        while routers or targets:
            rand_num = random.random()

            if (rand_num < self.__new_router_probability and targets) or not routers:
                row, col = targets.pop()
                yield Operator(True, row, col)
            else:
                row, col = routers.pop()
                yield Operator(False, row, col)

    def get_placement_neighborhood(self) -> Iterator[Operator]:
        targets = self.get_target_cells()
        random.shuffle(targets)

        for row, col in targets:
            yield Operator(True, row, col)

    def crossover(self, other: 'Building') -> Optional[Tuple['Building', 'Building']]:
        max_row, max_col = self.rows, self.columns

        lower_row = random.randint(0, max_row - 2)
        upper_row = random.randint(lower_row + 1, max_row - 1)
        lower_col = random.randint(0, max_col - 2)
        upper_col = random.randint(lower_col + 1, max_col - 1)

        def clear_edges(grid: CellArray) -> None:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            queue: Deque[Tuple[int, int]] = deque()

            for r in range(lower_row, upper_row):
                if (grid[r, lower_col] & self.BACKBONE_BIT) != 0:
                    queue.append((r, lower_col))
                if (grid[r, upper_col] & self.BACKBONE_BIT) != 0:
                    queue.append((r, upper_col))
            for c in range(lower_col, upper_col):
                if (grid[lower_row, c] & self.BACKBONE_BIT) != 0:
                    queue.append((lower_row, c))
                if (grid[upper_row, c] & self.BACKBONE_BIT) != 0:
                    queue.append((upper_row, c))

            while queue:
                r, c = queue.popleft()

                grid[r, c] &= ~(self.ROUTER_BIT | self.BACKBONE_BIT) \
                    if (r, c) != self.__backbone_root \
                    else ~self.ROUTER_BIT

                for dr, dc in directions:
                    new_r, new_c = r + dr, c + dc
                    if 0 <= new_r < max_row and 0 <= new_c < max_col and \
                        (grid[new_r, new_c] & self.BACKBONE_BIT) != 0 and \
                        (grid[new_r, new_c] & self.ROUTER_BIT) == 0 and \
                        (new_r, new_c) != self.__backbone_root:
                        queue.append((new_r, new_c))

        stripped_self = self.__cells & ~self.COVERED_BIT
        clear_edges(stripped_self)

        # pylint: disable=protected-access
        stripped_other = other.__cells & ~self.COVERED_BIT
        clear_edges(stripped_other)

        temp_rect = stripped_self[lower_row:upper_row, lower_col:upper_col].copy()
        stripped_self[lower_row:upper_row, lower_col:upper_col] = \
            stripped_other[lower_row:upper_row, lower_col:upper_col]
        stripped_other[lower_row:upper_row, lower_col:upper_col] = temp_rect

        assert self.problem is not None

        child1 = Building(stripped_self, self.__backbone_root,
                          self.__new_router_probability, self.problem)
        child1.reconnect_routers()
        if not self.problem.check_budget(child1):
            return None

        # pylint: disable=protected-access
        child2 = Building(stripped_other, self.__backbone_root,
                          self.__new_router_probability, self.problem)
        child2.reconnect_routers()
        if not self.problem.check_budget(child2):
            return None

        for router in child1.get_routers():
            child1.cover_neighbors(*router)
        for router in child2.get_routers():
            child2.cover_neighbors(*router)

        return child1, child2

    def is_same(self, other: 'Building') -> bool:
        # pylint: disable=protected-access
        return np.array_equal(self.__cells, other.__cells)

    def get_num_targets(self) -> int:
        return np.count_nonzero(self.__cells & self.CELL_TYPE_MASK == CellType.TARGET.value)

    @override
    def get_num_routers(self) -> int:
        return np.count_nonzero(self.__cells & self.ROUTER_BIT)

    @override
    def get_num_connected_cells(self) -> int:
        return np.count_nonzero(self.__cells & self.BACKBONE_BIT) - 1

    @override
    def get_coverage(self) -> int:
        return np.count_nonzero(self.__cells & (self.CELL_TYPE_MASK | self.COVERED_BIT)
                                == CellType.TARGET.value | self.COVERED_BIT)

    def get_num_uncovered_targets(self) -> int:
        return np.count_nonzero(self.__cells & (self.CELL_TYPE_MASK | self.COVERED_BIT)
                                == CellType.TARGET.value)

    @property
    def score(self) -> int:
        if self.__score is None:
            assert self.problem is not None
            self.__score = self.problem.get_score(self)
        return self.__score
