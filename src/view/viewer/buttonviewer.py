from abc import abstractmethod
from typing import Generic, Tuple, TypeVar

from src.view.viewer.pygameviewer import PygameViewer

T = TypeVar('T')
U = TypeVar('U')

class ButtonViewer(PygameViewer[T], Generic[T, U]):
    """
    A class that represents a button in a graphical interface, with functionality 
    for detecting clicks and handling button interactions.

    Args:
        T (TypeVar): The type of the entity the viewer works with.
        U (TypeVar): The type of the entity involved in the button's click action.
    """

    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initializes a ButtonViewer instance with position and size.

        Args:
            x (int): The x-coordinate of the top-left corner of the button.
            y (int): The y-coordinate of the top-left corner of the button.
            width (int): The width of the button.
            height (int): The height of the button.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def is_clicked(self, click_pos: Tuple[int, int]) -> bool:
        """
        Checks if the given position is within the button's area.

        Args:
            click_pos (Tuple[int, int]): The (x, y) position of the mouse click.

        Returns:
            bool: True if the click is within the button's bounds, otherwise False.
        """
        return self.x <= click_pos[0] <= self.x + self.width \
           and self.y <= click_pos[1] <= self.y + self.height

    @abstractmethod
    def on_click(self, entity: U) -> None:
        """
        An abstract method that defines the action to be performed when the button is clicked.
        Subclasses must implement this method.

        Args:
            entity (U): The entity associated with the button click action.
        """

    def handle_click(self, click_pos: Tuple[int, int], entity: U) -> None:
        """
        Handles a click event by checking if the click occurred within the button's area, 
        and if so, invokes the on_click method.

        Args:
            click_pos (Tuple[int, int]): The (x, y) position of the mouse click.
            entity (U): The entity associated with the button click action.
        """
        if self.is_clicked(click_pos):
            self.on_click(entity)

    @property
    def top_left_corner(self) -> Tuple[int, int]:
        """
        Returns the coordinates of the top-left corner of the button.

        Returns:
            Tuple[int, int]: The (x, y) coordinates of the button's top-left corner.
        """
        return (self.x, self.y)
