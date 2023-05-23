import numpy as np


class StaticOneD:
    # The 1D version of FreeGrad from [MK20]
    def __init__(self, eps, G):
        self.eps = eps
        self.V = G ** 2
        self.S = 0
        self.G = G

    def get_prediction(self):
        factor1 = np.exp(self.S ** 2 / 2 / (self.V + self.G * np.abs(self.S)))
        factor2 = (2 * self.V + self.G * np.abs(self.S)) / (self.V + self.G * np.abs(self.S)) ** 2
        factor3 = - self.eps * self.S * self.G ** 2 / 2 / np.sqrt(self.V)
        return factor1 * factor2 * factor3

    def update(self, gt):
        self.S += gt
        self.V += gt ** 2
