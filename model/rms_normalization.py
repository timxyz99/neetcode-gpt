import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        n = len(x)
        output = [0.0] * n

        sum_sqr = 0
        for i in range(n):
            sum_sqr += x[i] ** 2

        RMS = math.sqrt(sum_sqr / n + eps)

        for i in range(n):
            x_hat = x[i] / RMS
            output[i] = gamma[i] * x_hat

        return np.round(output, 4)