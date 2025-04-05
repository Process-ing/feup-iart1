from typing import Callable, override
import numpy as np

import pygame
from src.view.viewer.buttonviewer import ButtonViewer


class ChartButton(ButtonViewer[None, Callable[[], None]]):
    WIDTH = 48
    HEIGHT = 48
    ICON_WIDTH = 12
    ICON_HEIGHT = 12

    ICON = (np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]) * (256 ** 3 - 1)).transpose()

    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y, self.WIDTH, self.HEIGHT)

    @override
    def render(self, entity: None) -> pygame.Surface:
        screen = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        icon_surface = pygame.Surface((self.ICON_WIDTH, self.ICON_HEIGHT), pygame.SRCALPHA)
        for y in range(self.ICON_HEIGHT):
            for x in range(self.ICON_WIDTH):
                if self.ICON[x][y] != 0:
                    icon_surface.set_at((x, y), (255, 255, 255, 255))
                else:
                    icon_surface.set_at((x, y), (0, 0, 0, 0))

        icon_surface = pygame.transform.scale(icon_surface, (self.WIDTH, self.HEIGHT))
        screen.blit(icon_surface, (0, 0))

        return screen

    @override
    def on_click(self, toggle_graph: Callable[[], None]) -> None:
        toggle_graph()
