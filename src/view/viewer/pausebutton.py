from typing import Callable, override
import pygame
from src.view.viewer.buttonviewer import ButtonViewer


class PauseButton(ButtonViewer[bool, Callable[[], None]]):
    """
    A button that toggles between play and pause states. When clicked, it triggers
    the toggle_pause function passed to it.

    Attributes:
        WIDTH (int): The width of the pause button.
        HEIGHT (int): The height of the pause button.
        ICON_WIDTH (int): The width of the pause/play icon.
        ICON_HEIGHT (int): The height of the pause/play icon.
    """
    WIDTH = 48
    HEIGHT = 48
    ICON_WIDTH = 12
    ICON_HEIGHT = 12

    def __init__(self, x: int, y: int) -> None:
        """
        Initializes the PauseButton at the specified position.

        Args:
            x (int): The x-coordinate of the button's top-left corner.
            y (int): The y-coordinate of the button's top-left corner.
        """
        super().__init__(x, y, self.WIDTH, self.HEIGHT)

    @override
    def on_click(self, toggle_pause: Callable[[], None]) -> None:
        """
        Called when the button is clicked. It triggers the toggle_pause function.

        Args:
            toggle_pause (Callable[[], None]): A function to be called when the button is clicked
            to toggle the pause state.
        """
        toggle_pause()

    @override
    def render(self, paused: bool) -> pygame.Surface:
        """
        Renders the pause button's current state (play or pause).

        If the game is paused, a play icon (triangle) is drawn; otherwise, pause 
        icons (two vertical bars) are drawn.

        Args:
            paused (bool): The current pause state. If True, the game is paused; 
            otherwise, the game is playing.

        Returns:
            pygame.Surface: The surface representing the rendered button.
        """
        screen = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        screen.fill((0, 0, 0, 0))

        if not paused:
            pygame.draw.rect(screen, (255, 255, 255), (8, 8, 12, 32))
            pygame.draw.rect(screen, (255, 255, 255), (28, 8, 12, 32))
        else:
            triangle = pygame.Surface((self.ICON_WIDTH, self.ICON_HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(triangle, (255, 255, 255),
                [(2, 2), (3, 2), (9, 5), (9, 6), (3, 9), (2, 9)])
            triangle = pygame.transform.scale(triangle, (self.WIDTH, self.HEIGHT))
            screen.blit(triangle, (0, 0))

        return screen
