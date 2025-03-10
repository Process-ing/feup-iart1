from enum import Enum
from typing import Iterator, cast
import numpy as np

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

    def __init__(self, cells: CellArray):
        self.__cells: CellArray = cells

    @classmethod
    def from_text(cls, rows: int, columns: int, backbone: tuple[int, ...], text: str) -> 'Building':
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

        return cls(cells)

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

    # TODO(Process-ing): Remove this
    def place_router(self, row: int, column: int) -> None:
        self.__cells[row, column] = CellType.ROUTER.value

    def iter(self) -> Iterator[tuple[int, int, int]]:
        return ((row, column, int(cell)) for (row, column), cell in np.ndenumerate(self.__cells))
