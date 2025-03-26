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
        screen = pygame.Surface((self.WIDTH, self.HEIGHT))

        icon_screen = pygame.Surface((self.ICON_WIDTH, self.ICON_HEIGHT))
        pygame.pixelcopy.array_to_surface(icon_screen, self.ICON)
        icon_screen = pygame.transform.scale(icon_screen, (self.WIDTH, self.HEIGHT))

        screen.blit(icon_screen, (0, 0))

        return screen

    @override
    def on_click(self, toggle_graph: Callable[[], None]) -> None:
        toggle_graph()
        # TODO(Process-ing): Show score chart
