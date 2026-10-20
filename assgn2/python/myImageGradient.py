import numpy as np
import cv2


# Derivative kernels provided in the handout.
KX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64) / 8.0
KY = KX.T


def myImageGradient(img0):
    """Compute horizontal and vertical image derivatives.

    Parameters
    ----------
    img0 : ndarray, shape (H, W)
        Grayscale input image.

    Returns
    -------
    imgx, imgy : ndarray
        Horizontal and vertical derivatives.
    """
    img0 = np.asarray(img0, dtype=np.float64)
    if img0.ndim != 2:
        raise ValueError("img0 must be a 2D grayscale image")

    if img0.max() > 1.0:
        img0 = img0 / 255.0

    # TODO: filter img0 with KX and KY.
    # You may use cv2.filter2D or scipy.ndimage filtering.
    imgx = None
    imgy = None

    return imgx, imgy
