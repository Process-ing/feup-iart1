from typing import override
import pygame
from src.algorithm import Algorithm
from src.model import RouterProblem
from src.view.viewer import BuildingViewer
from src.view.window.pygamewindow import PygameWindow
from src.view.error import UnitializedError

class OptimizationWindow(PygameWindow):
    def __init__(self, problem: RouterProblem, algorithm: Algorithm,
        max_framerate: float = 0) -> None:

        super().__init__(max_framerate)
        self.__problem = problem
        self.__score = problem.get_score()
        self.__algorithm = algorithm
        self.__font: pygame.font.Font | None = None
        self.__building_viewer = BuildingViewer()

    @override
    def get_window_size(self) -> tuple[int, int]:
        building = self.__problem.building
        return self.__building_viewer.get_preferred_size(building)

    @override
    def get_window_caption(self) -> str:
        return 'Router Optimization'

    def on_init(self, screen: pygame.Surface) -> None:
        self.__font = pygame.font.Font('BigBlueTerm437NerdFont-Regular.ttf', 18)

    def __display(self, screen: pygame.Surface) -> None:
        problem_screen = self.__building_viewer.render(self.__problem.building)
        scaled_problem = pygame.transform.scale(problem_screen, screen.get_size())
        screen.blit(scaled_problem, (0, 0))

        if self.__font is None:
            raise UnitializedError('Font is not initialized')
        text = self.__font.render(f'Score: {self.__score}', True, (255, 255, 255))
        screen.blit(text, (10, 10))

    def on_update(self, screen: pygame.Surface) -> None:
        self.__display(screen)
        pygame.display.flip()

        self.__algorithm.step()
        self.__score = self.__problem.get_score()

    def pause(self) -> None:
        self.__algorithm.pause()
