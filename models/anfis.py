import numpy as np


class ANFISController:
    def __init__(self, n_inputs=8, n_rules=8):
        self.n_inputs = n_inputs
        self.n_rules = n_rules

        self.centers = np.random.randn(n_rules, n_inputs)
        self.widths = np.ones((n_rules, n_inputs))

        # 2-output linear consequents
        self.weights = np.random.randn(n_rules, n_inputs + 1, 2)

    def gaussian_mf(self, x, c, s):
        return np.exp(-((x - c) ** 2) / (2 * s ** 2))

    def compute_firing_strengths(self, x):
        firing = []
        for i in range(self.n_rules):
            mu = self.gaussian_mf(x, self.centers[i], self.widths[i])
            firing.append(np.prod(mu))
        firing = np.array(firing)
        firing /= np.sum(firing) + 1e-6
        return firing

    def forward(self, x):
        firing = self.compute_firing_strengths(x)
        x_ext = np.append(x, 1.0)

        output = np.zeros(2)
        for i in range(self.n_rules):
            output += firing[i] * (x_ext @ self.weights[i])

        return output
