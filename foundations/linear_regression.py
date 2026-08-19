import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        X = np.array(X)
        weights = np.array(weights)
        predictions = np.dot(X, weights)
        return np.round(predictions, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        model_prediction = np.array(model_prediction)
        ground_truth = np.array(ground_truth)
        n = len(model_prediction)
        mse = np.sum((model_prediction - ground_truth) ** 2) / n
        return round(mse, 5)
