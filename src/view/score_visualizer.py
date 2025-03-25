import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

class ScoreVisualizer:
    def __init__(self):
        self.scores = []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-')

    def update_scores(self, new_score):
        self.scores.append(new_score)
        self.update_plot()

    def update_plot(self):
        self.line.set_data(range(len(self.scores)), self.scores)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self):
        plt.ion()
        plt.show()


