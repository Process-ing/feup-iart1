from typing import List, Tuple, override
import pygame
from src.model import Building, CellType
from src.view.viewer.pygameviewer import PygameViewer


class BuildingViewer(PygameViewer[Building]):
    """
    Renders a Building object using a palette-based surface in pygame.
    This class is responsible for displaying the Building model in a 2D grid, 
    where each cell is rendered with different colors based on its type and state.
    """
    def __init__(self) -> None:
        """
        Initializes the BuildingViewer and sets up the color palette for rendering.
        """
        self.__palette = self.__create_palette()

    @override
    def render(self, entity: Building) -> pygame.Surface:
        """
        Renders the provided Building entity to a pygame surface.

        Args:
            entity (Building): The Building object to be rendered.

        Returns:
            pygame.Surface: A surface representing the Building object.
        """
        height, width = entity.shape

        # Create an 8-bit surface and set its color palette
        screen = pygame.Surface((width, height))
        screen = screen.convert(8)
        screen.set_palette(self.__palette)

        # Convert the Building entity to an array and transfer it to the surface
        array = entity.as_nparray_transposed()
        pygame.pixelcopy.array_to_surface(screen, array)

        return screen

    @staticmethod
    def __create_palette() -> List[Tuple[int, int, int]]:
        """
        Creates a color palette for rendering cells of the Building entity.
        Each cell type and state combination is assigned a unique color.

        Returns:
            List[Tuple[int, int, int]]: A list of RGB tuples representing the palette.
        """

        # Initialize a default palette with black (RGB 0, 0, 0)
        palette = [(0, 0, 0)] * 256

        # Assign colors to various cell types and their combinations
        palette[CellType.VOID.value] = (0x36, 0x00, 0x43)
        palette[CellType.VOID.value | Building.BACKBONE_BIT] = (0xec, 0xe2, 0x56)
        palette[CellType.VOID.value | Building.COVERED_BIT] = (0x36, 0x00, 0x43)
        palette[CellType.VOID.value | Building.BACKBONE_BIT | Building.COVERED_BIT] = \
            (0xec, 0xe2, 0x56)
        palette[CellType.VOID.value | Building.ROUTER_BIT | Building.BACKBONE_BIT \
                | Building.COVERED_BIT] = (0x7f, 0xc3, 0x82)

        palette[CellType.TARGET.value] = (0x20, 0x60, 0x71)
        palette[CellType.TARGET.value | Building.BACKBONE_BIT] = (0xec, 0xe2, 0x56)
        palette[CellType.TARGET.value | Building.COVERED_BIT] = (0x53, 0x92, 0xa4)
        palette[CellType.TARGET.value | Building.BACKBONE_BIT | Building.COVERED_BIT] = \
            (0xec, 0xe2, 0x56)
        palette[CellType.TARGET.value | Building.ROUTER_BIT | Building.BACKBONE_BIT \
                | Building.COVERED_BIT] = (0x7f, 0xc3, 0x82)

        palette[CellType.WALL.value] = (0x33, 0x35, 0x6c)
        palette[CellType.WALL.value | Building.BACKBONE_BIT] = (0xca, 0xb8, 0x1c)
        palette[CellType.WALL.value | Building.COVERED_BIT] = (0x33, 0x35, 0x6c)
        palette[CellType.WALL.value | Building.BACKBONE_BIT | Building.COVERED_BIT] = \
            (0xca, 0xb8, 0x1c)

        return palette

    def get_preferred_size(self, building: Building) -> Tuple[int, int]:
        """
        Calculates the preferred window size based on the Building dimensions 
        and the current screen size.

        Args:
            building (Building): The Building object for which the size is to be calculated.

        Returns:
            Tuple[int, int]: The preferred width and height of the window.
        """
        problem_height, problem_width = building.shape

        # Calculate cell size based on screen resolution and building dimensions
        cell_size = max(min(
            pygame.display.Info().current_w // problem_width,
            pygame.display.Info().current_h // problem_height
        ), 1)

        # Return the total window size based on cell size
        height, width = problem_height * cell_size, problem_width * cell_size

        return width, height
