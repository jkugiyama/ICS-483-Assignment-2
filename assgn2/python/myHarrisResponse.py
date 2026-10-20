import numpy as np
import cv2


def myHarrisResponse(imgx, imgy, sigma=1.5, k=0.04):
    """Compute the Harris response from image derivatives.

    Returns
    -------
    R, Sxx, Syy, Sxy : ndarray
        Harris response and smoothed gradient-product images.
    """
    imgx = np.asarray(imgx, dtype=np.float64)
    imgy = np.asarray(imgy, dtype=np.float64)

    if imgx.shape != imgy.shape:
        raise ValueError("imgx and imgy must have the same shape")

    # TODO 1: form the gradient products Ix^2, Iy^2, and Ix*Iy.
    Ix2 = imgx ** 2
    Iy2 = imgy ** 2
    IxIy = imgx * imgy

    # TODO 2: Gaussian smooth the three gradient products.
    Sxx = cv2.GaussianBlur(Ix2, (0, 0), sigma)
    Syy = cv2.GaussianBlur(Iy2, (0, 0), sigma)
    Sxy = cv2.GaussianBlur(IxIy, (0, 0), sigma)

    # TODO 3: compute
    # R = Sxx*Syy - Sxy**2 - k*(Sxx + Syy)**2
    R = Sxx * Syy - Sxy**2 - k * (Sxx + Syy)**2
    return R, Sxx, Syy, Sxy
