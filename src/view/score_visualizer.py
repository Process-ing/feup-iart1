import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


class ScoreVisualizer:
    def __init__(self):
        self.__scores = []
        self.__max_scores = []
        self.__enabled = False

    def update_scores(self, new_score):
        if self.__scores and new_score == self.__scores[-1]:
            return

        if not self.__max_scores or new_score > self.__max_scores[-1]:
            self.__max_scores.append(new_score)
        else:
            self.__max_scores.append(self.__max_scores[-1])

        self.__scores.append(new_score)
        self.update_plot()

    def update_plot(self):
        if not self.__enabled:
            return
        self.line.set_data(range(len(self.__scores)), self.__scores)
        self.max_line.set_data(range(len(self.__max_scores)), self.__max_scores)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def toggle_show_graph(self):
        if self.__enabled:
            plt.close(self.fig)
            self.__enabled = False
        else:
            self.__enabled = True
            self.fig, self.ax = plt.subplots()
            (self.line,) = self.ax.plot([], [], "r-", label="Score")
            (self.max_line,) = self.ax.plot([], [], "b--", label="Max Score")
            self.ax.legend()
            self.fig.canvas.manager.set_window_title("Score Graph")
            self.update_plot()
            self.show()

    def show(self):
        plt.ion()
        plt.show()
