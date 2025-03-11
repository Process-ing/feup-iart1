from typing import override

import pygame
from src.model import RouterProblem
from src.view.window.pygamewindow import PygameWindow
from src.view.viewer import BuildingViewer

class ProblemWindow(PygameWindow):
    def __init__(self, problem: RouterProblem) -> None:
        self.__problem = problem

    @override
    def get_window_size(self) -> tuple[int, int]:
        problem_height, problem_width = self.__problem.building.shape
        cell_size = max(min(
            pygame.display.Info().current_w // problem_width,
            pygame.display.Info().current_h // problem_height
        ), 1)
        height, width = problem_height * cell_size, problem_width * cell_size

        return width, height

    @override
    def get_window_caption(self) -> str:
        return 'Router Problem'

    @override
    def on_init(self) -> None:
        self.__building_viewer = BuildingViewer()

    @override
    def on_display(self, screen: pygame.Surface) -> None:
        problem_screen = self.__building_viewer.render(self.__problem.building)
        scaled_problem = pygame.transform.scale(problem_screen, screen.get_size())
        screen.blit(scaled_problem, (0, 0))

    @override
    def on_update(self) -> None:
        pass
