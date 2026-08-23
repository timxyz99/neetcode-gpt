import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        n = len(x)
        u = (1/n) * np.sum(x)
        var = (1/n) * np.sum((x - u) ** 2)
        x_hat = [0] * n
        for i in range(n):
            x_hat[i] = ((x[i] - u) / math.sqrt(var + 1e-5)) * gamma[i] + beta[i]
        
        return np.round(x_hat, 5)
