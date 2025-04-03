from typing import List
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


class ScoreVisualizer:
    def __init__(self) -> None:
        self.__scores: List[float] = []
        self.__max_scores: List[float] = []
        self.__enabled = False

        self.__fig, self.__ax = plt.subplots()
        self.__line = self.__ax.plot([], [], "r-", label="Score")[0]
        self.__max_line = self.__ax.plot([], [], "b--", label="Max Score")[0]
        self.__ax.legend()
        manager = self.__fig.canvas.manager
        assert manager is not None
        manager.set_window_title("Score Graph")

    def update_scores(self, new_score: float) -> None:
        if self.__scores and new_score == self.__scores[-1]:
            return

        if not self.__max_scores or new_score > self.__max_scores[-1]:
            self.__max_scores.append(new_score)
        else:
            self.__max_scores.append(self.__max_scores[-1])

        self.__scores.append(new_score)
        self.update_plot()

    def update_plot(self) -> None:
        if not self.__enabled:
            return

        self.__line.set_data(range(len(self.__scores)), self.__scores)
        self.__max_line.set_data(range(len(self.__max_scores)), self.__max_scores)
        self.__ax.relim()
        self.__ax.autoscale_view()
        self.__fig.canvas.draw()
        self.__fig.canvas.flush_events()

    def toggle_show_graph(self) -> None:
        if self.__enabled:
            plt.close(self.__fig)
            self.__enabled = False
        else:
            self.__enabled = True
            self.update_plot()
            self.show()

    def show(self) -> None:
        plt.ion()
        plt.show()

    def cleanup(self) -> None:
        if self.__enabled:
            plt.close(self.__fig)
