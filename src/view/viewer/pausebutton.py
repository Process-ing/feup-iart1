from typing import Callable, override
import pygame
from src.view.viewer.buttonviewer import ButtonViewer


class PauseButton(ButtonViewer[bool, Callable[[], None]]):
    WIDTH = 48
    HEIGHT = 48
    ICON_WIDTH = 12
    ICON_HEIGHT = 12

    def __init__(self, x: int, y: int) -> None:
        super().__init__(x, y, self.WIDTH, self.HEIGHT)

    @override
    def on_click(self, toggle_pause: Callable[[], None]) -> None:
        toggle_pause()

    @override
    def render(self, paused: bool) -> pygame.Surface:
        screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        screen.fill((0, 0, 0))

        if not paused:
            pygame.draw.rect(screen, (255, 255, 255), (8, 8, 12, 32))
            pygame.draw.rect(screen, (255, 255, 255), (28, 8, 12, 32))
        else:
            triangle = pygame.Surface((self.ICON_WIDTH, self.ICON_HEIGHT))
            pygame.draw.polygon(triangle, (255, 255, 255),
                [(2, 2), (3, 2), (9, 5), (9, 6), (3, 9), (2, 9)])
            triangle = pygame.transform.scale(triangle, (self.WIDTH, self.HEIGHT))
            screen.blit(triangle, (0, 0))

        return screen
