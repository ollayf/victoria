from utils.sounds import usual
from scipy import stats
import numpy as np
import utils

class ModeCounter:

    def __init__(self, master, max_values=None) -> None:
        self.stream = master.stream
        self.last_mode = 0
        self.master = master
        if not max_values:
            max_values = self.rec_max_value()
        self.values = np.zeros(max_values)

    def rec_max_value(self):
        fps = self.stream.fps
        return int(13*fps//20)

    def update(self, value):
        self.values = np.roll(self.values, -1)
        self.values[-1] = value
        new_mode = stats.mode(self.values)[0][0]
        if new_mode != self.last_mode:
            if new_mode > self.last_mode:
                print("Mode person added")
            else:
                print("Mode person removed")
            self.last_mode = new_mode
            usual()
        return new_mode