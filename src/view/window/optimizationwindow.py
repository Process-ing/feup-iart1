from typing import List, Optional, Tuple, override
from threading import Thread, Event, get_ident
import time
import pygame

from src.algorithm import Algorithm
from src.model import RouterProblem
from src.view.viewer import BuildingViewer, PauseButton
from src.view.window.pygamewindow import PygameWindow
from src.view.viewer import ChartButton, ChartViewer

class OptimizationWindow(PygameWindow):
    """
    A Pygame window that runs and visualizes the optimization process of router placement
    using a given algorithm on a building model.
    """
    def __init__(self, problem: RouterProblem, algorithm: Algorithm,
                 max_framerate: float = 60) -> None:
        """
        Initialize the optimization window with the problem and algorithm.

        Args:
            problem (RouterProblem): The router placement problem instance.
            algorithm (Algorithm): The optimization algorithm to be run.
            max_framerate (float): The maximum frame rate of the window.
        """

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
        self.__font: Optional[pygame.font.Font] = None
        self.__building_viewer = BuildingViewer()
        self.__show_chart = False
        self.__pause_button: Optional[PauseButton] = None
        self.__chart_button: Optional[ChartButton] = None
        self.__chart_viewer: ChartViewer

        self.__execution_thread: Optional[Thread] = None
        self.__continue_event = Event()
        self.__continue_event.set()
        self.__stop_execution = False

    @override
    def get_window_size(self) -> Tuple[int, int]:
        """
        Get the preferred window size based on the building dimensions.

        Returns:
            Tuple[int, int]: Width and height of the window.
        """
        building = self.__problem.building
        return self.__building_viewer.get_preferred_size(building)

    @override
    def get_window_caption(self) -> str:
        """
        Get the caption for the window.

        Returns:
            str: The window caption.
        """
        return 'Router Optimization'

    def get_best_score(self) -> int:
        """
        Get the best score achieved during optimization.

        Returns:
            int: The highest score achieved.
        """
        return self.__max_score

    def get_duration(self) -> float:
        """
        Get the total duration of the optimization process.

        Returns:
            float: Time in seconds from start to end of execution.
        """
        return self.__end_time - self.__start_time

    def __run_algorithm(self) -> None:
        """
        Run the optimization algorithm in a separate thread, updating
        the visualization and statistics throughout execution.
        """
        self.__start_time = time.perf_counter()
        self.__chart_viewer.set_start_time()
        self.__chart_viewer.add_score(self.__score)

        for information_message in self.__algorithm.run():
            if self.__stop_execution:
                break

            self.__continue_event.wait()

            # Sets the information message (bottom right corner)
            if information_message is not None:
                self.__information_message = information_message

            # Updates the score and other information (top left corner)
            self.__score = self.__problem.building.score
            self.__max_score = max(self.__max_score, self.__score)
            self.__num_covered_cells = self.__problem.building.get_coverage()
            self.__num_routers = self.__problem.building.get_num_routers()
            self.__chart_viewer.add_score(self.__score)

        self.__problem.building = self.__problem.best_building
        self.__information_message = 'Done!'
        self.__end_time = time.perf_counter()

    def on_init(self, screen: pygame.Surface) -> None:
        """
        Initialize the window components and start the optimization thread.

        Args:
            screen (pygame.Surface): The surface to render the window on.
        """
        width, height = self.get_window_size()
        self.__pause_button = PauseButton(width - (48 + 8) * 2, 8)
        self.__chart_button = ChartButton(width - (48 + 8), 8)
        self.__font = pygame.font.Font('BigBlueTerm437NerdFont-Regular.ttf', 20)

        self.__chart_viewer = ChartViewer(width, height)

        self.__execution_thread = Thread(target=self.__run_algorithm)
        self.__execution_thread.start()

    def __draw_info_message(self, screen: pygame.Surface) -> None:
        """
        Draw a temporary information message on the screen.

        Args:
            screen (pygame.Surface): The surface to render on.
        """
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
        """
        Draw score, number of routers, and coverage statistics on the screen.

        Args:
            screen (pygame.Surface): The surface to render on.
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


    def __draw_buttons(self, screen: pygame.Surface) -> None:
        """
        Draw pause and chart toggle buttons on the screen.

        Args:
            screen (pygame.Surface): The surface to render on.
        """
        assert self.__pause_button is not None
        assert self.__chart_button is not None

        screen_width = screen.get_size()[0]
        quad_width = 2 * (12 + 2) + 3
        quad_height = 12 + 2 + 3

        quad = pygame.Surface((quad_width, quad_height), pygame.SRCALPHA)
        pygame.draw.polygon(quad, (0, 0, 0, 128), [
            (0, 0), (quad_width - 1, 0), (quad_width - 1, quad_height - 1),
            (3, quad_height - 1), (0, quad_height - 4)
        ])

        quad = pygame.transform.scale(quad, (quad_width * 4, quad_height * 4))
        screen.blit(quad, (screen_width - quad_width * 4, 0))

        chart_button_screen = self.__chart_button.render(None)
        screen.blit(chart_button_screen, self.__chart_button.top_left_corner)

        pause_button_screen = self.__pause_button.render(not self.__continue_event.is_set())
        screen.blit(pause_button_screen, self.__pause_button.top_left_corner)

    def __display(self, screen: pygame.Surface) -> None:
        """
        Handle rendering logic depending on the current display mode
        (building view or chart).

        Args:
            screen (pygame.Surface): The surface to render on.
        """

        # Optimization thread might send an update before fully pausing
        if not self.__show_chart and self.__continue_event.is_set():
            problem_screen = self.__building_viewer.render(self.__problem.building)
            scaled_problem = pygame.transform.scale(problem_screen, screen.get_size())
            screen.blit(scaled_problem, (0, 0))
            self.__draw_info_message(screen)

        if self.__show_chart:
            chart = self.__chart_viewer.render(None)
            screen.blit(chart, (0, 0))

        self.__draw_info(screen)
        self.__draw_buttons(screen)


    def on_update(self, events: List[pygame.event.Event], screen: pygame.Surface) -> None:
        """
        Process input events and update the display accordingly.

        Args:
            events (List[pygame.event.Event]): A list of Pygame events.
            screen (pygame.Surface): The surface to render on.
        """
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
        """
        Toggle the execution pause state of the optimization algorithm.
        """
        if self.__continue_event.is_set():
            self.__continue_event.clear()
        else:
            self.__continue_event.set()

    def toggle_show_chart(self) -> None:
        """
        Toggle the display mode between building view and chart.
        """
        self.__show_chart = not self.__show_chart

    def pause(self) -> None:
        """
        Pause the optimization process.
        """
        self.__continue_event.clear()

    def cleanup(self) -> None:
        """
        Stop the optimization thread and clean up before exiting.
        """
        assert self.__execution_thread is not None

        self.__stop_execution = True
        self.__continue_event.set()

        if self.__execution_thread.ident != get_ident():
            self.__execution_thread.join()

            if self.__end_time == 0:
                self.__end_time = time.perf_counter()

    def dump_to_file(self, filename: str) -> None:
        """
        Save the results of the optimization to a text file.

        Args:
            filename (str): The output filename.
        """
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f'Score: {self.__max_score}\n')
            file.write(f'Time: {self.__end_time - self.__start_time} s\n')
