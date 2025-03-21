from enum import Enum
from typing import Iterator, Tuple, cast
from collections import deque
import numpy as np
from copy import deepcopy

from src.model.error import ProblemLoadError

type CellArray = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]

class CellType(Enum):
    VOID = 0
    TARGET = 1
    WALL = 2

class Building:
    BACKBONE_BIT = 1 << 7  # Marks a cell connected to the backbone
    COVERED_BIT = 1 << 6  # Marks a cell covered by a router
    ROUTER_BIT = 1 << 5  # Marks a cell as a router
    CELL_TYPE_MASK = 0b00011111  # Mask for the cell type

    CHAR_CELL_MAP = {
        ord('-'): CellType.VOID,
        ord('.'): CellType.TARGET,
        ord('#'): CellType.WALL,
    }

    def __init__(self, cells: CellArray, router_range: int) -> None:
        self.__cells: CellArray = cells
        self.__router_range = router_range

    @classmethod
    def from_text(cls, shape: Tuple[int, int], backbone: tuple[int, ...],
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

        return cls(cells, router_range)

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

    def connect_neighbors(self, row: int, col: int) -> None:
        R = self.__router_range

        row_start = max(0, row - R)
        row_len = min(self.__cells.shape[0] - row_start, 2 * R + 1)
        col_start = max(0, col - R)
        col_len = min(self.__cells.shape[1] - col_start, 2 * R + 1)

        ctr_row = row - row_start
        ctr_col = col - col_start

        neighborhood = np.zeros((row_len, col_len), dtype=np.uint8)
        neighborhood |= self.__cells[row_start:row_start + row_len, col_start:col_start + col_len] & Building.CELL_TYPE_MASK

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
            (((nrow, ncol) for nrow in range(ctr_row - 1, -1, -1) for ncol in range(ctr_col - 1, -1, -1)), 1, 1),
            (((nrow, ncol) for nrow in range(ctr_row - 1, -1, -1) for ncol in range(ctr_col + 1, col_len)), 1, -1),
            (((nrow, ncol) for nrow in range(ctr_row + 1, row_len) for ncol in range(ctr_col - 1, -1, -1)), -1, 1),
            (((nrow, ncol) for nrow in range(ctr_row + 1, row_len) for ncol in range(ctr_col + 1, col_len)), -1, -1),
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
        # Check if position is valid
        # current_cell = self.__cells[row, column]

        if self.__cells[row, column] & self.CELL_TYPE_MASK == CellType.WALL.value:
            return False

        # # Check if router is already placed
        if (self.__cells[row, column] & self.ROUTER_BIT) != 0:
            return False

        self.__cells[row, column] |= self.ROUTER_BIT | self.COVERED_BIT | self.BACKBONE_BIT
        self.connect_neighbors(row, column)
        return True

        # TODO(henriquesfernandes): Connect router to the backbone


        #
        # if self.__cells[row, column] & self.CELL_TYPE_MASK != CellType.WALL.value:
        #     self.__cells[row, column] = CellType.ROUTER.value
        #     # NOTE(Process-ing): This is not checking walls while specifying connectivity!
        #     self.__cells[row - self.__router_range:row + self.__router_range + 1,
        #                  column - self.__router_range:column + self.__router_range + 1] \
        #         |= self.COVERED_BIT

        # return self.copy()

    def remove_router(self, row: int, column: int) -> bool:
        # Ignore if router is already placed
        if (self.__cells[row, column] & self.ROUTER_BIT) == 0:
            return False

        self.__cells[row, column] &= ~self.ROUTER_BIT
        return True

    def get_neighborhood(self) -> Iterator['Building']:
        for row in range(self.__cells.shape[0]):
            for col in range(self.__cells.shape[1]):
                neighbor = deepcopy(neighbor)
                if neighbor.place_router(row, col):
                    yield neighbor
                elif neighbor.remove_router(row, col):
                    yield neighbor
