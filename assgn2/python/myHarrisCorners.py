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

    # TODO: threshold R and apply non-maximum suppression.
    # You may use scipy.ndimage.maximum_filter.
    # Return an N x 2 array in [x, y] order.
    
    R = np.asarray(R, dtype=np.float64)

    if R.ndim != 2:
        raise ValueError("R must be a 2D response map")
    if not np.all(np.isfinite(R)):
        raise ValueError("R must contain only finite values")
    if not np.isfinite(thresholdRel) or thresholdRel < 0:
        raise ValueError("thresholdRel must be finite and non-negative")
    if not isinstance(nmsSize, (int, np.integer)) or nmsSize < 1:
        raise ValueError("nmsSize must be a positive integer")

    max_response = np.max(R)
    if max_response <= 0:
        return np.empty((0, 2), dtype=int)

    threshold = thresholdRel * max_response
    local_max = maximum_filter(R, size=nmsSize, mode="constant", cval=-np.inf)
    y, x = np.nonzero((R > threshold) & (R == local_max))
    corners = np.column_stack((x, y))

    return corners
