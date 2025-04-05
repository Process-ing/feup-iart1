from typing import List, Optional, Tuple, override
from threading import Thread, Event, get_ident
import time
import pygame

from src.algorithm import Algorithm
from src.model import RouterProblem
from src.view.viewer import BuildingViewer, PauseButton
from src.view.window.pygamewindow import PygameWindow
from src.view.viewer import ChartButton
from src.view.score_visualizer import ScoreVisualizer

class OptimizationWindow(PygameWindow):
    def __init__(self, problem: RouterProblem, algorithm: Algorithm,
                 score_visualizer: ScoreVisualizer, max_framerate: float = 60) -> None:

        super().__init__(max_framerate)

        self.__problem = problem
        self.__score = problem.building.score
        self.__num_covered_cells = problem.building.get_coverage()
        self.__num_routers = problem.building.get_num_routers()
        self.__max_score = problem.building.score
        self.__start_time: float = 0
        self.__end_time: float = 0
        self.__information_message = ''
        self.__algorithm = algorithm
        self.__score_visualizer = score_visualizer
        self.__font: Optional[pygame.font.Font] = None
        self.__building_viewer = BuildingViewer()
        self.__pause_button: Optional[PauseButton] = None
        self.__chart_button: Optional[ChartButton] = None

        self.__execution_thread: Optional[Thread] = None
        self.__continue_event = Event()
        self.__continue_event.set()
        self.__stop_execution = False

    @override
    def get_window_size(self) -> Tuple[int, int]:
        building = self.__problem.building
        return self.__building_viewer.get_preferred_size(building)

    @override
    def get_window_caption(self) -> str:
        return 'Router Optimization'

    def get_best_score(self) -> int:
        return self.__max_score

    def get_duration(self) -> float:
        return self.__end_time - self.__start_time

    def __run_algorithm(self) -> None:
        self.__start_time = time.perf_counter()
        for information_message in self.__algorithm.run():
            if self.__stop_execution:
                break

            self.__continue_event.wait()

            if information_message is not None:
                self.__information_message = information_message
            self.__score = self.__problem.building.score
            self.__max_score = max(self.__max_score, self.__score)
            self.__num_covered_cells = self.__problem.building.get_coverage()
            self.__num_routers = self.__problem.building.get_num_routers()
            self.__score_visualizer.update_scores(self.__score)

        self.__problem.building = self.__problem.best_building
        self.__information_message = 'Done!'
        self.__end_time = time.perf_counter()

    def on_init(self, screen: pygame.Surface) -> None:
        width = self.get_window_size()[0]
        self.__pause_button = PauseButton(width - (48 + 8) * 2, 8)
        self.__chart_button = ChartButton(width - (48 + 8), 8)
        self.__font = pygame.font.Font('BigBlueTerm437NerdFont-Regular.ttf', 20)

        self.__execution_thread = Thread(target=self.__run_algorithm)
        self.__execution_thread.start()

    def __draw_info_message(self, screen: pygame.Surface) -> None:
        assert self.__font is not None

        message_text = self.__font.render(self.__information_message, True, (255, 255, 255))
        message_width = message_text.get_width()
        message_height = message_text.get_height()
        message_screen = pygame.Surface((message_width + 20, message_height + 10), pygame.SRCALPHA)
        pygame.draw.polygon(message_screen, (0, 0, 0, 128), [
            (0, 0), (message_width + 19, 0), (message_width + 19, message_height + 9),
            (message_width + 14, message_height + 9), (0, message_height + 9)
        ])
        screen_width, screen_height = screen.get_size()
        screen.blit(message_screen, (screen_width - message_width - 20,
                                     screen_height - message_height - 10))
        screen.blit(message_text, (screen_width - message_width - 10,
                                   screen_height - message_height - 5))

    def __draw_info(self, screen: pygame.Surface) -> None:
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

        pause_button_screen = self.__pause_button.render(not self.__continue_event.is_set())
        screen.blit(pause_button_screen, self.__pause_button.top_left_corner)

    def __display(self, screen: pygame.Surface) -> None:
        # Optimization thread might send an update before fully pausing
        if self.__continue_event.is_set():
            problem_screen = self.__building_viewer.render(self.__problem.building)
            scaled_problem = pygame.transform.scale(problem_screen, screen.get_size())
            screen.blit(scaled_problem, (0, 0))

        self.__draw_info(screen)
        if self.__continue_event.is_set():
            self.__draw_info_message(screen)
        self.__draw_buttons(screen)

    def on_update(self, events: List[pygame.event.Event], screen: pygame.Surface) -> None:
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                assert self.__pause_button is not None
                assert self.__chart_button is not None

                click_pos = pygame.mouse.get_pos()
                self.__pause_button.handle_click(click_pos, self.toggle_pause)
                self.__chart_button.handle_click(click_pos, self.toggle_show_chart)

        self.__display(screen)
        pygame.display.flip()

    def toggle_pause(self) -> None:
        if self.__continue_event.is_set():
            self.__continue_event.clear()
        else:
            self.__continue_event.set()

    def toggle_show_chart(self) -> None:
        self.__score_visualizer.toggle_show_chart()

    def pause(self) -> None:
        self.__continue_event.clear()

    def cleanup(self) -> None:
        assert self.__execution_thread is not None

        self.__stop_execution = True
        self.__continue_event.set()

        self.__score_visualizer.cleanup()
        if self.__execution_thread.ident != get_ident():
            self.__execution_thread.join()

            if self.__end_time == 0:
                self.__end_time = time.perf_counter()

    def dump_to_file(self, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f'Score: {self.__max_score}\n')
            file.write(f'Time: {self.__end_time - self.__start_time} s\n')
