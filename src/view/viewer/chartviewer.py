import time
from typing import List
import pygame
import pygame_chart as pyc
from src.view.viewer.pygameviewer import PygameViewer


class ChartViewer(PygameViewer[None]):
    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self.__scores: List[float] = []
        self.__max_scores: List[float] = []
        self.__times: List[float] = []
        self.__start_time = 0.0

        self.__screen = pygame.Surface((width, height))
        self.__figure = pyc.Figure(self.__screen, 0, 0, width, height)
        self.__figure.add_title('Score Chart')
        self.__figure.add_legend()
        self.__figure.add_xaxis_label('Time (s)')
        self.__figure.add_yaxis_label('Score')

    def set_start_time(self) -> None:
        """
        Set the start time for the chart.
        """
        self.__start_time = time.perf_counter()

    def add_score(self, score: float) -> None:
        # Optimization to reduce the number of points
        if len(self.__scores) >= 2 and score == self.__scores[-1] and score == self.__scores[-2]:
            self.__scores.pop()
            self.__max_scores.pop()
            self.__times.pop()

        self.__scores.append(score)
        self.__max_scores.append(max(score, self.__max_scores[-1] if self.__max_scores else -1))
        elapsed_time = time.perf_counter() - self.__start_time
        self.__times.append(elapsed_time)

    def render(self, _: None) -> pygame.Surface:
        if len(self.__scores) >= 2:
            self.__figure.line('Score', self.__times, self.__scores, color=(255, 0, 0))
            self.__figure.line('Max Score', self.__times, self.__max_scores, color=(0, 0, 255))

            self.__figure.draw()
        return self.__screen
