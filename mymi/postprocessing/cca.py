import numpy as np
from skimage.measure import label   

def get_largest_cc(a: np.ndarray) -> np.ndarray:
    """
    returns: a 3D array with largest connected component only.
    args:
        a: a 3D binary array.
    """
    if a.dtype != np.bool:
        raise ValueError(f"'get_batch_largest_cc' expected a boolean array, got '{a.dtype}'.")

    # Check that there's at least 1 connected component.
    labels = label(a)
    if labels.max() == 0:
        raise ValueError(f"No foreground pixels found when calculating 'get_largest_cc'.")
    
    # Calculate largest component.
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largest_cc

def get_batch_largest_cc(a: np.ndarray) -> np.ndarray:
    """
    returns: a batch of 3D arrays with largest connected component only.
    args:
        a: a 3D binary array.
    """
    if a.dtype != np.bool:
        raise ValueError(f"'get_batch_largest_cc' expected a boolean array, got '{a.dtype}'.")

    b = np.zeros_like(a)
    for i, data in enumerate(a):
        largest_cc = get_largest_cc([i])
        cc = get_largest_cc(data)
        b[i] = cc

    return b
        