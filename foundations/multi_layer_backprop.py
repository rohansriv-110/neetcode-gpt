import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        n = len(y_true)

        # --- forward ---
        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        y_hat = z2

        loss = np.mean((y_hat - y_true) ** 2)

        # --- backward ---
        dz2 = 2 * (y_hat - y_true) / n
        dW2 = np.outer(dz2, a1)
        db2 = dz2

        da1 = dz2 @ W2
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }
