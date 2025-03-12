from abc import abstractmethod
from typing import Generic, Tuple, TypeVar

from src.view.viewer.pygameviewer import PygameViewer

T = TypeVar('T')
U = TypeVar('U')

class ButtonViewer(PygameViewer[T], Generic[T, U]):
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def is_clicked(self, click_pos: Tuple[int, int]) -> bool:
        return self.x <= click_pos[0] <= self.x + self.width \
           and self.y <= click_pos[1] <= self.y + self.height

    @abstractmethod
    def on_click(self, entity: U) -> None:
        pass

    def handle_click(self, click_pos: Tuple[int, int], entity: U) -> None:
        if self.is_clicked(click_pos):
            self.on_click(entity)

    @property
    def top_left_corner(self) -> Tuple[int, int]:
        return (self.x, self.y)
