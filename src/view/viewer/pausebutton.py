from typing import Callable
import pygame
from src.view.viewer.buttonviewer import ButtonViewer


class PauseButton(ButtonViewer[None]):
    def __init__(self, x: int, y: int, toggle_pause: Callable[[], None]) -> None:
        super().__init__(x, y, 48, 48)

        self.toggle_pause = toggle_pause

    def on_click(self, entity: None) -> None:
        self.toggle_pause()

    def render(self, entity: None):
        screen = pygame.Surface((self.width, self.height))
        screen.fill((255, 255, 255))
        return screen
