from typing import Tuple, override
import pygame
from src.model import Building, CellType
from src.view.viewer.pygameviewer import PygameViewer


class BuildingViewer(PygameViewer[Building]):
    def __init__(self) -> None:
        self.__palette = self.__create_pallete()

    @override
    def render(self, entity: Building) -> pygame.Surface:
        height, width = entity.shape

        screen = pygame.Surface((width, height))
        screen = screen.convert(8)
        screen.set_palette(self.__palette)

        array = entity.as_nparray_transposed()
        pygame.pixelcopy.array_to_surface(screen, array)

        return screen

    @staticmethod
    def __create_pallete() -> list[Tuple[int, int, int]]:
        palette = [(0, 0, 0)] * 256

        palette[CellType.VOID.value] = (0x36, 0x00, 0x43)
        palette[CellType.VOID.value | Building.BACKBONE_BIT] = (0xec, 0xe2, 0x56)
        palette[CellType.VOID.value | Building.COVERED_BIT] = (0x36, 0x00, 0x43)
        palette[CellType.VOID.value | Building.BACKBONE_BIT | Building.COVERED_BIT] = \
            (0xec, 0xe2, 0x56)
        palette[CellType.VOID.value | Building.ROUTER_BIT | Building.BACKBONE_BIT \
                | Building.COVERED_BIT] = (0x7f, 0xc3, 0x82)

        palette[CellType.TARGET.value] = (0x20, 0x60, 0x71)
        palette[CellType.TARGET.value | Building.BACKBONE_BIT] = (0xec, 0xe2, 0x56)
        palette[CellType.TARGET.value | Building.COVERED_BIT] = (0x53, 0x92, 0xa4)
        palette[CellType.TARGET.value | Building.BACKBONE_BIT | Building.COVERED_BIT] = \
            (0xec, 0xe2, 0x56)
        palette[CellType.TARGET.value | Building.ROUTER_BIT | Building.BACKBONE_BIT \
                | Building.COVERED_BIT] = (0x7f, 0xc3, 0x82)

        palette[CellType.WALL.value] = (0x33, 0x35, 0x6c)
        palette[CellType.WALL.value | Building.BACKBONE_BIT] = (0xca, 0xb8, 0x1c)
        palette[CellType.WALL.value | Building.COVERED_BIT] = (0x33, 0x35, 0x6c)
        palette[CellType.WALL.value | Building.BACKBONE_BIT | Building.COVERED_BIT] = \
            (0xca, 0xb8, 0x1c)

        return palette

    def get_preferred_size(self, building: Building) -> Tuple[int, int]:
        problem_height, problem_width = building.shape
        cell_size = max(min(
            pygame.display.Info().current_w // problem_width,
            pygame.display.Info().current_h // problem_height
        ), 1)
        height, width = problem_height * cell_size, problem_width * cell_size

        return width, height
