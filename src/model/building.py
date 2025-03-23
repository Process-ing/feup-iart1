from enum import Enum
from typing import Iterator, Tuple, cast
from collections import deque
from copy import deepcopy
import numpy as np
import random

from src.model.disjoint_set import DisjointSet
from src.model.error import ProblemLoadError

type CellArray = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]

class CellType(Enum):
    VOID = 0
    TARGET = 1
    WALL = 2

class Building:
    BACKBONE_BIT = np.uint8(1 << 7)  # Marks a cell connected to the backbone
    COVERED_BIT = np.uint8(1 << 6)  # Marks a cell covered by a router
    ROUTER_BIT = np.uint8(1 << 5)  # Marks a cell as a router
    CELL_TYPE_MASK = np.uint8(0b00011111)  # Mask for the cell type

    CHAR_CELL_MAP = {
        ord('-'): CellType.VOID,
        ord('.'): CellType.TARGET,
        ord('#'): CellType.WALL,
    }

    def __init__(self, cells: CellArray, router_range: int, backbone: tuple[int, int]) -> None:
        self.__cells: CellArray = cells
        self.__router_range = router_range
        self.__backbone_root = backbone

    @classmethod
    def from_text(cls, shape: Tuple[int, int], backbone: tuple[int, int],
                  text: str, router_range: int) -> 'Building':
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

        return cls(cells, router_range, backbone)

    @property
    def rows(self) -> int:
        return self.__cells.shape[0]

    @property
    def columns(self) -> int:
        return self.__cells.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.__cells.shape)

    def __str__(self) -> str:
        return '\n'.join(''.join(map(chr, row)) for row in self.__cells)

    def as_nparray(self) -> CellArray:
        return self.__cells.copy()

    def as_nparray_transposed(self) -> CellArray:
        return self.__cells.transpose()

    def iter(self) -> Iterator[tuple[int, int, int]]:
        return ((row, column, int(cell)) for (row, column), cell in np.ndenumerate(self.__cells))

    def get_connected_routers(self, root: tuple[int, int]) \
        -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
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
        rrange = self.__router_range

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

    # TODO(Process-ing): Remove this
    def place_router(self, row: int, column: int) -> bool:

        current_cell = self.__cells[row, column]

        # Check if position is valid (routers cannot be placed inside walls)
        if current_cell & self.CELL_TYPE_MASK == CellType.WALL.value: #or current_cell & self.CELL_TYPE_MASK == CellType.VOID.value:
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
        parent = {}

        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1),(-1, 0), (1, 0), (0, -1), (0, 1) ]

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
        cell_row_start = max(0, row - self.__router_range)
        cell_row_end = min(self.__cells.shape[0], row + self.__router_range + 1)
        cell_col_start = max(0, column - self.__router_range)
        cell_col_end = min(self.__cells.shape[1], column + self.__router_range + 1)

        self.__cells[cell_row_start:cell_row_end, cell_col_start:cell_col_end] &= ~self.COVERED_BIT

        router_row_start = max(0, row - 2 * self.__router_range)
        router_row_end = min(self.__cells.shape[0], row + 2 * self.__router_range + 1)
        router_col_start = max(0, column - 2 * self.__router_range)
        router_col_end = min(self.__cells.shape[1], column + 2 * self.__router_range + 1)

        for rrow in range(router_row_start, router_row_end):
            for rcol in range(router_col_start, router_col_end):
                if self.__cells[rrow, rcol] & self.ROUTER_BIT:
                    self.__cells[rrow, rcol] |= self.COVERED_BIT
                    self.cover_neighbors(rrow, rcol)

    def remove_router(self, row: int, column: int) -> bool:
        # Ignore if router is already placed
        if (self.__cells[row, column] & self.ROUTER_BIT) == 0:
            return False

        self.__cells[row, column] &= ~(self.ROUTER_BIT | self.BACKBONE_BIT) \
            if (self.__cells[row, column] & self.BACKBONE_BIT) == 0 \
            else ~self.ROUTER_BIT
        
        self.update_neighbor_coverage(row, column)
        self.reconnect_routers()
        return True

    def reconnect_routers(self) -> None:
        self.__cells &= ~self.BACKBONE_BIT
        self.__cells[self.__backbone_root] |= self.BACKBONE_BIT

        routers = list(zip(*np.where(self.__cells & self.ROUTER_BIT)))
        if not routers:
            return

        def reconstruct_path(pred, p: tuple[int, int]) -> set[tuple[int, int]]:
            res = set()
            while p:
                res.add(p)
                p = pred.get(p, None)
            return res

        def steiner_tree(grid, terminals: list[tuple[int, int]]) -> set[tuple[int, int]]:
            rows, cols = len(grid), len(grid[0])
            directions = [(1, 1), (-1, -1), (1, -1), (-1, 1), (1, 0), (-1, 0), (0, 1), (0, -1)]

            terminal_indices = {t: i for i, t in enumerate(terminals)}
            dsu = DisjointSet(len(terminals))
            queue = deque()
            source = {}
            pred = {}
            res = set()

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

    def get_neighborhood(self) -> Iterator['Building']:
        while True:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.columns - 1)

            neighbor: Building = deepcopy(self)
            if neighbor.place_router(row, col):
                yield neighbor
            elif neighbor.remove_router(row, col):
                yield neighbor

    def lazy_next_move(self, place_probability: float) -> Iterator[Tuple[str, int, int]]:
        used = set()

        while True:
            if random.random() < place_probability:
                while True:
                    row = random.randint(0, self.rows - 1)
                    col = random.randint(0, self.columns - 1)
                    if (row, col) not in used:
                        used.add((row, col))
                        print("Placing router at", row, col)
                        yield ('place', row, col)
                        break;
            else:
                # TODO(henriquesfernandes): Implement router removal
                print("Removing router")
                yield ('remove', 0, 0)

    def get_num_routers(self) -> int:
        return np.count_nonzero(self.__cells & self.ROUTER_BIT)

    def get_num_connected_cells(self) -> int:
        return np.count_nonzero(self.__cells & self.BACKBONE_BIT) - 1

    def get_coverage(self) -> int:
        return np.count_nonzero(self.__cells & (self.CELL_TYPE_MASK | self.COVERED_BIT) == CellType.TARGET.value | self.COVERED_BIT)