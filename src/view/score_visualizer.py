import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

class ScoreVisualizer:
    def __init__(self):
        self.scores = []
        self.max_scores = []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-', label='Score')
        self.max_line, = self.ax.plot([], [], 'b-', label='Max Score')
        self.ax.legend()

    def update_scores(self, new_score):
        if (self.scores and new_score == self.scores[-1]):
            return

        if (not self.max_scores or new_score > self.max_scores[-1]):
            self.max_scores.append(new_score)
        else:
            self.max_scores.append(self.max_scores[-1])


        self.scores.append(new_score)
        self.update_plot()

    def update_plot(self):
        self.line.set_data(range(len(self.scores)), self.scores)
        self.max_line.set_data(range(len(self.max_scores)), self.max_scores)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self):
        plt.ion()
        plt.show()


