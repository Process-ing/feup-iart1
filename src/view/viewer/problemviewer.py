from typing import override
import pygame
from src.model import Building, RouterProblem, CellType
from src.view.viewer.pygameviewer import PygameViewer


class ProblemViewer(PygameViewer[RouterProblem]):
    def __init__(self) -> None:
        self.PALETTE = self.__create_pallete()

    @override
    def render(self, problem: RouterProblem) -> pygame.Surface:
        height, width = problem.building.shape

        screen = pygame.Surface((width, height))
        screen = screen.convert(8)
        screen.set_palette(self.PALETTE)

        array = problem.building.as_nparray_transposed()
        pygame.pixelcopy.array_to_surface(screen, array)

        return screen

    @staticmethod
    def __create_pallete() -> list[tuple[int, int, int]]:
        palette = [(0, 0, 0)] * 256

        palette[CellType.VOID.value] = (0x36, 0x00, 0x43)
        palette[CellType.VOID.value | Building.BACKBONE_BIT] = (0xec, 0xe2, 0x56)
        palette[CellType.VOID.value | Building.CONNECTED_BIT] = (0x36, 0x00, 0x43)
        palette[CellType.VOID.value | Building.BACKBONE_BIT | Building.CONNECTED_BIT] = \
            (0xec, 0xe2, 0x56)

        palette[CellType.TARGET.value] = (0x20, 0x60, 0x71)
        palette[CellType.TARGET.value | Building.BACKBONE_BIT] = (0xec, 0xe2, 0x56)
        palette[CellType.TARGET.value | Building.CONNECTED_BIT] = (0x53, 0x92, 0xa4)
        palette[CellType.TARGET.value | Building.BACKBONE_BIT | Building.CONNECTED_BIT] = \
            (0xec, 0xe2, 0x56)

        palette[CellType.WALL.value] = (0x33, 0x35, 0x6c)
        palette[CellType.WALL.value | Building.BACKBONE_BIT] = (0xca, 0xb8, 0x1c)
        palette[CellType.WALL.value | Building.CONNECTED_BIT] = (0x33, 0x35, 0x6c)
        palette[CellType.WALL.value | Building.BACKBONE_BIT | Building.CONNECTED_BIT] = \
            (0xca, 0xb8, 0x1c)

        palette[CellType.ROUTER.value] = (0x7f, 0xc3, 0x82)
        palette[CellType.ROUTER.value | Building.BACKBONE_BIT] = (0x7f, 0xc3, 0x82)
        palette[CellType.ROUTER.value | Building.CONNECTED_BIT] = (0x7f, 0xc3, 0x82)
        palette[CellType.ROUTER.value | Building.BACKBONE_BIT | Building.CONNECTED_BIT] = \
            (0x7f, 0xc3, 0x82)

        return palette