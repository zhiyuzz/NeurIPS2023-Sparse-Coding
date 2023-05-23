import numpy as np


class DynamicBaseline:
    def __init__(self, eps, G, T):
        self.G = G
        self.Max_k = int(np.ceil(np.log2(np.sqrt(T))))
        self.eps = eps / self.Max_k
        self.V = 4 * G ** 2

        self.step_sizes = np.zeros(self.Max_k)
        self.w = np.zeros(self.Max_k)
        self.theta = np.zeros(self.Max_k)
        for k in range(self.Max_k):
            self.step_sizes[k] = np.minimum(2 ** (k + 1) / np.sqrt(T), 1) / G

        self.alpha = self.eps * (self.G ** 2) / self.V / (np.log(self.V / self.G ** 2) ** 2)

    def get_prediction(self):
        return np.sum(self.w)

    def update(self, gt):
        for k in range(self.Max_k):
            self.theta[k] = 2 / self.step_sizes[k] * np.sign(self.w[k]) * np.log(1 + np.abs(self.w[k]) / self.alpha) - gt

        self.V += gt ** 2
        self.alpha = self.eps * (self.G ** 2) / self.V / (np.log(self.V / self.G ** 2) ** 2)

        for k in range(self.Max_k):
            self.w[k] = self.alpha * np.sign(self.theta[k]) * (np.exp(self.step_sizes[k] / 2 * np.maximum(np.abs(self.theta[k]) - 2 * self.step_sizes[k] * gt ** 2, 0)) - 1)
