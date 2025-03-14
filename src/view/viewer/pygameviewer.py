from abc import abstractmethod
from typing import Generic, TypeVar
import pygame

T = TypeVar('T')

class PygameViewer(Generic[T]):
    @abstractmethod
    def render(self, entity: T) -> pygame.Surface:
        pass
