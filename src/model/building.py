import numpy as np

type CellArray = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]

class Building:
    BACKBONE_MASK = 1 << 7
    # TODO(Process-ing): Map characters it if ends up being worth it

    def __init__(self, cells: CellArray):
        self.__cells: CellArray = cells

    @classmethod
    def from_text(cls, rows: int, columns: int, backbone: tuple[int, ...], text: str) -> 'Building':
        cells = np.frombuffer(text.replace('\n', '').encode(), dtype=np.uint8)
        cells = cells.reshape((rows, columns)).copy()

        cells[backbone] = int.from_bytes(b'b') | cls.BACKBONE_MASK

        return cls(cells)

    def __getitem__(self, key: tuple[int | slice, int | slice]) -> CellArray:
        return self.__cells[key]

    def __str__(self) -> str:
        return '\n'.join(''.join(map(chr, row)) for row in self.__cells)
