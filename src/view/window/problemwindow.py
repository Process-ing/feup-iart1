from typing import List, Optional, Tuple, override

import pygame
from src.model import RouterProblem
from src.view.window.pygamewindow import PygameWindow
from src.view.viewer import BuildingViewer

class ProblemWindow(PygameWindow):
    """
    A Pygame window that displays the state of a RouterProblem without allowing interaction.
    Shows the building layout, score, number of routers, and covered cells.
    """
    def __init__(self, problem: RouterProblem) -> None:
        """
        Initializes the ProblemWindow.

        Args:
            problem (RouterProblem): The router optimization problem instance.
        """
        super().__init__(max_framerate=60)  # A low framerate slows down window exit
        self.__problem = problem
        self.__score = problem.building.score
        self.__num_covered_cells = problem.building.get_coverage()
        self.__num_routers = problem.building.get_num_routers()
        self.__font: Optional[pygame.font.Font] = None
        self.__building_viewer = BuildingViewer()

    @override
    def get_window_size(self) -> Tuple[int, int]:
        """
        Returns the preferred size of the window based on the building layout.

        Returns:
            Tuple[int, int]: Width and height of the window.
        """
        building = self.__problem.building
        return self.__building_viewer.get_preferred_size(building)

    @override
    def get_window_caption(self) -> str:
        """
        Returns the caption to display on the window.

        Returns:
            str: The window caption.
        """
        return 'Router Problem'

    @override
    def on_init(self, screen: pygame.Surface) -> None:
        """
        Called once during initialization. Draws the building and score information.

        Args:
            screen (pygame.Surface): The surface to draw on.
        """
        problem_screen = self.__building_viewer.render(self.__problem.building)
        scaled_problem = pygame.transform.scale(problem_screen, screen.get_size())
        self.__font = pygame.font.Font('BigBlueTerm437NerdFont-Regular.ttf', 20)

        screen.blit(scaled_problem, (0, 0))
        self.__draw_info(screen)
        pygame.display.flip()

    def __draw_info(self, screen: pygame.Surface) -> None:
        """
        Draws the score, number of routers, and coverage information on the screen.

        Args:
            screen (pygame.Surface): The surface to draw the text overlay on.
        """
        assert self.__font is not None

        score_text = self.__font.render(f'Score: {self.__score}',
                                          True, (255, 255, 255))
        routers_text = self.__font.render(f'Routers: {self.__num_routers}',
                                          True, (255, 255, 255))
        covered_text = self.__font.render(f'Covered: {self.__num_covered_cells}',
                                          True, (255, 255, 255))

        text_width = max(
            score_text.get_width(),
            routers_text.get_width(),
            covered_text.get_width()
        )

        text_height = (
            score_text.get_height() +
            routers_text.get_height() +
            covered_text.get_height() +
            20
        )
        info_width = text_width + 20
        info_height = text_height + 10

        info_screen = pygame.Surface((info_width, info_height), pygame.SRCALPHA)
        pygame.draw.polygon(info_screen, (0, 0, 0, 128), [
            (0, 0), (info_width - 1, 0), (info_width - 1, info_height - 4),
            (info_width - 4, info_height - 1), (0, info_height - 1)
        ])

        screen.blit(info_screen, (0, 0))

        screen.blit(score_text, (10, 10))
        screen.blit(routers_text, (10, 10 + score_text.get_height() + 5))
        screen.blit(
            covered_text,
            (
                10,
                10 + score_text.get_height() + routers_text.get_height() + 10
            )
        )

    @override
    def on_update(self, events: List[pygame.event.Event], screen: pygame.Surface) -> None:
        """
        Handles events and updates each frame. This window is static, so it does nothing.

        Args:
            events (List[pygame.event.Event]): The list of events since the last frame.
            screen (pygame.Surface): The surface to draw the updated frame on.
        """
        pass
