from typing import Self
import numpy as np

class Building:
    def __init__(self, cells: np.ndarray):
        self.__cells: np.ndarray = cells

    @staticmethod
    def from_text(rows: int, columns: int, text: str) -> Self:
        cells = np.fromstring(text.replace('\n', ''), dtype=np.uint8)
        cells = cells.reshape((rows, columns))

        return Building(cells)

    def __getitem__(self, key: tuple[int | slice, int | slice]):
        return self.__cells[key[0], key[1]]

    def __setitem__(self, key: tuple[int | slice, int | slice], new_cell: str):
        self.__cells[key[0], key[1]] = new_cell
