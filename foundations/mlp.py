import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        h=x
        last=len(weights)-1
        for i in range(len(weights)):
            z=h@weights[i] + biases[i]
            if i< last:
                h=np.maximum(0,z)
            else:
                h=z
        return np.round(h,5)

        
