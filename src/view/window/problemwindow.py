from typing import override

import pygame
from src.model import RouterProblem
from src.view.window.pygamewindow import PygameWindow
from src.view.viewer import BuildingViewer

class ProblemWindow(PygameWindow):
    def __init__(self, problem: RouterProblem) -> None:
        super().__init__(max_framerate=60)  # A low framerate slows down window exit
        self.__problem = problem
        self.__building_viewer = BuildingViewer()

    @override
    def get_window_size(self) -> tuple[int, int]:
        building = self.__problem.building
        return self.__building_viewer.get_preferred_size(building)

    @override
    def get_window_caption(self) -> str:
        return 'Router Problem'

    @override
    def on_init(self, screen: pygame.Surface) -> None:
        problem_screen = self.__building_viewer.render(self.__problem.building)
        scaled_problem = pygame.transform.scale(problem_screen, screen.get_size())

        screen.blit(scaled_problem, (0, 0))
        pygame.display.flip()

    @override
    def on_update(self, screen: pygame.Surface) -> None:
        pass
