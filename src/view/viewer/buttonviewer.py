from abc import abstractmethod
from typing import Generic, Tuple, TypeVar

import pygame
from src.view.viewer.pygameviewer import PygameViewer


T = TypeVar('T')

class ButtonViewer(PygameViewer[None], Generic[T]):
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def is_clicked(self, click_pos: Tuple[int, int]) -> bool:
        return self.x <= click_pos[0] <= self.x + self.width and self.y <= click_pos[1] <= self.y + self.height

    @abstractmethod
    def on_click(self, entity: T) -> None:
        pass

    def handle_click(self, event: pygame.event.Event, entity: T) -> None:
        if self.is_clicked(event):
            self.on_click(entity)

    @property
    def topLeftCorner(self) -> Tuple[int, int]:
        return (self.x, self.y)