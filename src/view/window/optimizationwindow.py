from typing import override
import pygame
from src.algorithm import Algorithm
from src.model import RouterProblem
from src.view.viewer import BuildingViewer
from src.view.window.pygamewindow import PygameWindow

class OptimizationWindow(PygameWindow):
    def __init__(self, problem: RouterProblem, algorithm: Algorithm,
        max_framerate: float = 0) -> None:

        super().__init__(max_framerate)
        self.__problem = problem
        self.__score = problem.get_score()
        self.__algorithm = algorithm

    def get_window_size(self) -> tuple[int, int]:
        problem_height, problem_width = self.__problem.building.shape
        cell_size = max(min(
            pygame.display.Info().current_w // problem_width,
            pygame.display.Info().current_h // problem_height
        ), 1)
        height, width = problem_height * cell_size, problem_width * cell_size

        return width, height

    def get_window_caption(self) -> str:
        return 'Router Optimization'

    def on_init(self) -> None:
        self.__font = pygame.font.Font('BigBlueTerm437NerdFont-Regular.ttf', 18)
        self.__building_viewer = BuildingViewer()

    def on_display(self, screen: pygame.Surface) -> None:
        problem_screen = self.__building_viewer.render(self.__problem.building)
        scaled_problem = pygame.transform.scale(problem_screen, screen.get_size())
        screen.blit(scaled_problem, (0, 0))

        text = self.__font.render(f'Score: {self.__score}', True, (255, 255, 255))
        screen.blit(text, (10, 10))

    def on_update(self) -> None:
        self.__algorithm.step()
        self.__score = self.__problem.get_score()
