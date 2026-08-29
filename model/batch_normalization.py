import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        batch_size = len(x)
        num_features = len(x[0])
        y = [[0.0] * num_features for i in range(batch_size)]

        if training:
            for i in range(num_features):
                
                sum_feature = 0
                for j in range(batch_size):
                    sum_feature += x[j][i]
                u = sum_feature / batch_size

                sum_sqr = 0
                for j in range(batch_size):
                    sum_sqr += (x[j][i] - u) ** 2
                variance = sum_sqr / batch_size

                for j in range(batch_size):
                    x_hat = (x[j][i] - u) / math.sqrt(variance + eps)
                    y[j][i] = gamma[i] * x_hat + beta[i]

                running_mean[i] = (1 - momentum) * running_mean[i] + momentum * u
                running_var[i] = (1 - momentum) * running_var[i] + momentum * variance
        else:

            for i in range(num_features):
                for j in range(batch_size):
                    x_hat = (x[j][i] - running_mean[i]) / math.sqrt(running_var[i] + eps)
                    y[j][i] = x_hat * gamma[i] + beta[i]

        y_rounded = np.round(y, 4).tolist()
        RN_rounded = np.round(running_mean, 4).tolist()
        RV_rounded = np.round(running_var, 4).tolist()

        return (y_rounded, RN_rounded, RV_rounded)
