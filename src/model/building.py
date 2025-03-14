from enum import Enum
from typing import Iterator, Tuple, cast
import numpy as np
from collections import deque

from src.model.error import ProblemLoadError

type CellArray = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]

class CellType(Enum):
    VOID = 0
    TARGET = 1
    WALL = 2
    ROUTER = 3

class Building:
    BACKBONE_BIT = 1 << 7  # Marks a cell connected to the backbone
    CONNECTED_BIT = 1 << 6  # Marks a cell connected to a router
    CELL_TYPE_MASK = 0b00111111  # Mask for the cell type

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

    # TODO(Process-ing): Remove this
    def place_router(self, row: int, column: int) -> None:
        if self.__cells[row, column] & self.CELL_TYPE_MASK != CellType.WALL.value:
            self.__cells[row, column] = CellType.ROUTER.value
            # NOTE(Process-ing): This is not checking walls while specifying connectivity!
            self.__cells[row - self.__router_range:row + self.__router_range + 1,
                         column - self.__router_range:column + self.__router_range + 1] \
                |= self.CONNECTED_BIT

    def iter(self) -> Iterator[tuple[int, int, int]]:
        return ((row, column, int(cell)) for (row, column), cell in np.ndenumerate(self.__cells))
    
    def get_connected_routers(self, root):
        routers = set()
        visited = set()
        queue = deque([root])
        visited.add(root)
        directions = [0, 1, 0, -1, 0]

        while queue:
            row, col = queue.popleft()
            if self.__cells[row, col] == CellType.ROUTER.value:
                routers.add((row,col))
            for i in range(4):
                nr, nc = row + directions[i], col + directions[i+1]
                if 0 <= nr < self.rows and 0 <= nc < self.columns and \
                        (nr, nc) not in visited and \
                        self.__cells[nr, nc] & self.BACKBONE_BIT:
                    queue.append((nr, nc))
                    visited.add((nr, nc))
        return routers


