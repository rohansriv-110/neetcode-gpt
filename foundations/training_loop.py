import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        n = len(y)
        w = np.zeros(X.shape[1])
        b = 0.0

        for _ in range(epochs):
            error = X @ w + b - y
            dw = (2/n) * X.T @ error
            db = (2/n) * np.sum(error)
            w = w - lr * dw
            b = b - lr * db

        return (np.round(w, 5), round(b, 5))