from abc import abstractmethod
from typing import Generic, TypeVar
import pygame

T = TypeVar('T')

class PygameViewer(Generic[T]):
    """
    An abstract base class for viewers that render a specific entity using Pygame.

    This class serves as a blueprint for creating custom viewers that display different
    types of entities (e.g., buttons, charts) using the Pygame library. The `render` method
    must be implemented by subclasses to define how to render the given entity.

    Attributes:
        T (TypeVar): A generic type that represents the type of entity this viewer will render.
    """
    @abstractmethod
    def render(self, entity: T) -> pygame.Surface:
        """
        Renders the given entity and returns a Pygame surface.

        Subclasses must implement this method to specify how the entity should be rendered.

        Args:
            entity (T): The entity to be rendered, which can be any type based on the generic `T`.

        Returns:
            pygame.Surface: The Pygame surface containing the rendered entity.
        """
