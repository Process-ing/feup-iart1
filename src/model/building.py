from typing import Self
import numpy as np

class Building:
    def __init__(self, cells: np.char.chararray):
        self.__cells = cells

    @staticmethod
    def from_text(rows: int, columns: int, text: str) -> Self:
        cells = np.char.chararray(rows, columns)
        cells[:, :] = [list(line) for line in text.split('\n')]

        return Building(cells)

    def __getitem__(self, key: tuple[int | slice, int | slice]):
        return self.__cells[key[0], key[1]]

    def __setitem__(self, key: tuple[int | slice, int | slice], new_cell: str):
        self.__cells[key[0], key[1]] = new_cell
