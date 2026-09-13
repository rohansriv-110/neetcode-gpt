import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        mu=np.mean(x)
        var=np.var(x)
        eps=1e-5
        x_norm=(x-mu)/np.sqrt(var+eps)
        out = x_norm * gamma + beta
        
        return np.round(out, 5)


        
        





