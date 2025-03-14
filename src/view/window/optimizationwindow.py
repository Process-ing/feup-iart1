from typing import List, override
import pygame
from src.algorithm import Algorithm
from src.model import RouterProblem
from src.view.viewer import BuildingViewer, PauseButton
from src.view.window.pygamewindow import PygameWindow
from src.view.error import UnitializedError
from src.view.viewer import ChartButton

class OptimizationWindow(PygameWindow):
    def __init__(self, problem: RouterProblem, algorithm: Algorithm,
        max_framerate: float = 0) -> None:

        super().__init__(max_framerate)

        self.__problem = problem
        self.__score = problem.get_score()
        self.__algorithm = algorithm
        self.__font: pygame.font.Font | None = None
        self.__building_viewer = BuildingViewer()
        self.__pause_button: PauseButton | None = None
        self.__chart_button: ChartButton | None = None
        self.__paused = False

    @override
    def get_window_size(self) -> tuple[int, int]:
        building = self.__problem.building
        return self.__building_viewer.get_preferred_size(building)

    @override
    def get_window_caption(self) -> str:
        return 'Router Optimization'

    def on_init(self, screen: pygame.Surface) -> None:
        width = self.get_window_size()[0]
        self.__pause_button = PauseButton(width - (48 + 8) * 2, 8)
        self.__chart_button = ChartButton(width - (48 + 8), 8)
        self.__font = pygame.font.Font('BigBlueTerm437NerdFont-Regular.ttf', 20)

    def __draw_info(self, screen: pygame.Surface) -> None:
        assert self.__font is not None

        INFO_WIDTH = 50
        INFO_HEIGHT = 12

        info_screen = pygame.Surface((INFO_WIDTH, INFO_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(info_screen, (0, 0, 0, 128), [
            (0, 0), (INFO_WIDTH - 1, 0), (INFO_WIDTH - 1, INFO_HEIGHT - 4),
            (INFO_WIDTH - 4, INFO_HEIGHT - 1), (0, INFO_HEIGHT - 1)
        ])
        info_screen = pygame.transform.scale(info_screen, (INFO_WIDTH * 4, INFO_HEIGHT * 4))

        text = self.__font.render(f'Score: {self.__score}', True, (255, 255, 255))

        screen.blit(info_screen, (0, 0))
        screen.blit(text, (10, 10))

    def __draw_buttons(self, screen: pygame.Surface) -> None:
        assert self.__pause_button is not None
        assert self.__chart_button is not None

        screen_width = screen.get_size()[0]
        QUAD_WIDTH = 2 * (12 + 2) + 3
        QUAD_HEIGHT = 12 + 2 + 3

        quad = pygame.Surface((QUAD_WIDTH, QUAD_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(quad, (0, 0, 0), [
            (0, 0), (QUAD_WIDTH - 1, 0), (QUAD_WIDTH - 1, QUAD_HEIGHT - 1),
            (3, QUAD_HEIGHT - 1), (0, QUAD_HEIGHT - 4)
        ])
        pygame.draw.lines(quad, (255, 255, 255), False, [
            (QUAD_WIDTH - 1, QUAD_HEIGHT - 1), (3, QUAD_HEIGHT - 1),
            (0, QUAD_HEIGHT - 4), (0, 0)
        ])

        quad = pygame.transform.scale(quad, (QUAD_WIDTH * 4, QUAD_HEIGHT * 4))
        screen.blit(quad, (screen_width - QUAD_WIDTH * 4, 0))

        chart_button_screen = self.__chart_button.render(None)
        screen.blit(chart_button_screen, self.__chart_button.top_left_corner)

        pause_button_screen = self.__pause_button.render(self.__paused)
        screen.blit(pause_button_screen, self.__pause_button.top_left_corner)

    def __display(self, screen: pygame.Surface) -> None:
        problem_screen = self.__building_viewer.render(self.__problem.building)
        scaled_problem = pygame.transform.scale(problem_screen, screen.get_size())
        screen.blit(scaled_problem, (0, 0))

        self.__draw_info(screen)
        self.__draw_buttons(screen)

    def on_update(self, events: List[pygame.event.Event], screen: pygame.Surface) -> None:
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                assert self.__pause_button is not None
                assert self.__chart_button is not None

                click_pos = pygame.mouse.get_pos()
                self.__pause_button.handle_click(click_pos, self.toggle_pause)
                self.__chart_button.handle_click(click_pos, self.pause)

        self.__display(screen)
        pygame.display.flip()

        if self.__paused:
            return

        self.__algorithm.step()
        self.__score = self.__problem.get_score()

    def toggle_pause(self) -> None:
        self.__paused = not self.__paused

    def pause(self) -> None:
        self.__paused = True
