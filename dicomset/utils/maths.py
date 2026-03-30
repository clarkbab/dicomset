import numpy as np
from typing import List

from ..typing import Number
from .conversion import to_list, to_numpy

def round(
    x: Number | List[Number] | np.ndarray,
    tol: Number = 1.0,
    ) -> Number | List[Number] | np.ndarray:
    x, return_type = to_numpy(x, return_type=True)
    x = tol * np.round(x / tol)
    if return_type is int or return_type is float:
        return return_type(x[0])
    elif return_type is list:
        return to_list(x)
    else:
        return x