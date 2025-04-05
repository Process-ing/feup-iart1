import time
from typing import List
import pygame
import pygame_chart as pyc
from src.view.viewer.pygameviewer import PygameViewer


class ChartViewer(PygameViewer[None]):
    """
    A class that displays a real-time chart of scores over time using Pygame.

    This class inherits from PygameViewer and provides functionality to track and 
    display scores, their maximum values, and the time elapsed since the start of 
    the chart. It uses the `pygame_chart` library to render the chart and update it 
    dynamically as new scores are added.

    Attributes:
        __scores (List[float]): A list of the scores added to the chart.
        __max_scores (List[float]): A list of the maximum scores at each point in time.
        __times (List[float]): A list of the elapsed time for each score.
        __start_time (float): The time when the chart started, used to calculate elapsed time.
        __screen (pygame.Surface): The surface on which the chart is drawn.
        __figure (pyc.Figure): The figure representing the chart, including axes, title, and lines.
    """
    def __init__(self, width: int, height: int) -> None:
        """
        Initializes the ChartViewer with the given width and height.
        """
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
        Sets the start time for the chart. This is used to calculate the elapsed time 
        for plotting the scores over time.

        This method is intended to be called before any scores are added.
        """
        self.__start_time = time.perf_counter()

    def add_score(self, score: float) -> None:
        """
        Adds a new score to the chart. This method also updates the maximum score 
        and the elapsed time for the current score.

        Optimizes the chart by removing duplicate consecutive scores.

        Args:
            score (float): The new score to be added to the chart.
        """

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
        """
        Renders the chart by drawing the score and maximum score over time.

        Args:
            _ (None): This argument is required by the PygameViewer interface,
            but it's not used in this method.

        Returns:
            pygame.Surface: The surface that contains the rendered chart.
        """
        if len(self.__scores) >= 2:
            self.__figure.line('Score', self.__times, self.__scores, color=(255, 0, 0))
            self.__figure.line('Max Score', self.__times, self.__max_scores, color=(0, 0, 255))

            self.__figure.draw()
        return self.__screen
