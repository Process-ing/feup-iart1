from typing import Any
import numpy as np

class Building:
    def __init__(self, cells: np.ndarray[Any, np.dtype[np.uint8]]):
        self.__cells: np.ndarray[Any, np.dtype[np.uint8]] = cells

    @classmethod
    def from_text(cls, rows: int, columns: int, text: str) -> 'Building':
        cells = np.frombuffer(text.replace('\n', '').encode(), dtype=np.uint8)
        cells = cells.reshape((rows, columns))

        return Building(cells)

    def __getitem__(self, key: tuple[int | slice, int | slice]):
        return self.__cells[key[0], key[1]]

    def __setitem__(self, key: tuple[int | slice, int | slice], new_cell: str):
        self.__cells[key[0], key[1]] = new_cell
