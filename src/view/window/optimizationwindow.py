from typing import List, override
import pygame
from src.algorithm import Algorithm
from src.model import RouterProblem
from src.view.viewer import BuildingViewer, PauseButton
from src.view.window.pygamewindow import PygameWindow
from src.view.viewer import ChartButton
from src.view.score_visualizer import ScoreVisualizer

class OptimizationWindow(PygameWindow):
    def __init__(self, problem: RouterProblem, algorithm: Algorithm, visualizer: ScoreVisualizer,
        max_framerate: float = 60) -> None:

        super().__init__(max_framerate)

        self.__problem = problem
        self.__score = problem.get_score(problem.building)
        self.__algorithm = algorithm
        self.__visualizer = visualizer
        self.__run = algorithm.run()
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

        info_width = 50
        info_height = 12

        info_screen = pygame.Surface((info_width, info_height), pygame.SRCALPHA)
        pygame.draw.polygon(info_screen, (0, 0, 0, 128), [
            (0, 0), (info_width - 1, 0), (info_width - 1, info_height - 4),
            (info_width - 4, info_height - 1), (0, info_height - 1)
        ])
        info_screen = pygame.transform.scale(info_screen, (info_width * 4, info_height * 4))

        text = self.__font.render(f'Score: {self.__score}', True, (255, 255, 255))

        screen.blit(info_screen, (0, 0))
        screen.blit(text, (10, 10))

    def __draw_buttons(self, screen: pygame.Surface) -> None:
        assert self.__pause_button is not None
        assert self.__chart_button is not None

        screen_width = screen.get_size()[0]
        quad_width = 2 * (12 + 2) + 3
        quad_height = 12 + 2 + 3

        quad = pygame.Surface((quad_width, quad_height), pygame.SRCALPHA)
        pygame.draw.polygon(quad, (0, 0, 0), [
            (0, 0), (quad_width - 1, 0), (quad_width - 1, quad_height - 1),
            (3, quad_height - 1), (0, quad_height - 4)
        ])
        pygame.draw.lines(quad, (255, 255, 255), False, [
            (quad_width - 1, quad_height - 1), (3, quad_height - 1),
            (0, quad_height - 4), (0, 0)
        ])

        quad = pygame.transform.scale(quad, (quad_width * 4, quad_height * 4))
        screen.blit(quad, (screen_width - quad_width * 4, 0))

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

        next(self.__run, None)
        self.__score = self.__problem.get_score(self.__problem.building)
        self.__visualizer.update_scores(self.__score)

    def toggle_pause(self) -> None:
        self.__paused = not self.__paused

    def pause(self) -> None:
        self.__paused = True
