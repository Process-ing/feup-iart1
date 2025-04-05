from typing import Callable, override
import numpy as np

import pygame
from src.view.viewer.buttonviewer import ButtonViewer


class ChartButton(ButtonViewer[None, Callable[[], None]]):
    """
    A button that represents a chart icon and triggers an action when clicked.

    Attributes:
        WIDTH (int): The width of the button.
        HEIGHT (int): The height of the button.
        ICON_WIDTH (int): The width of the chart icon.
        ICON_HEIGHT (int): The height of the chart icon.
        ICON (np.ndarray): The icon representing the chart,
        a 2D array of values where non-zero values
        are rendered as white pixels.
    """

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
        """
        Initializes a new ChartButton at the given position.

        Args:
            x (int): The x-coordinate of the button.
            y (int): The y-coordinate of the button.
        """
        super().__init__(x, y, self.WIDTH, self.HEIGHT)

    @override
    def render(self, entity: None) -> pygame.Surface:
        """
        Renders the chart icon onto a surface.

        Args:
            entity (None): This argument is unused in this method,
            as no specific entity is being rendered.

        Returns:
            pygame.Surface: The surface containing the rendered chart icon.
        """
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
        """
        Executes the provided callable function when the button is clicked.

        Args:
            toggle_graph (Callable[[], None]): A callable function that
            is triggered when the button is clicked.
        """
        toggle_graph()
