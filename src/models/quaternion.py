import numpy as np


class Quaternion:

    def __init__(self, s, x, y, z):
        self.s = s
        self.v = np.array([x, y, z])

