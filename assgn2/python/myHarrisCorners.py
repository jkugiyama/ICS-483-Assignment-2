import numpy as np
from scipy.ndimage import maximum_filter


def myHarrisCorners(R, thresholdRel=0.01, nmsSize=7):
    """Select strong local maxima from a Harris response map.

    Parameters
    ----------
    R : ndarray, shape (H, W)
        Harris response map.
    thresholdRel : float
        Relative threshold as a fraction of max(R).
    nmsSize : int
        Width/height of the local NMS neighborhood.

    Returns
    -------
    corners : ndarray, shape (N, 2)
        Detected [x, y] corner coordinates.
    """
    R = np.asarray(R, dtype=np.float64)

    # TODO: threshold R and apply non-maximum suppression.
    # You may use scipy.ndimage.maximum_filter.
    # Return an N x 2 array in [x, y] order.
    corners = None

    return corners
