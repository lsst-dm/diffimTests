import numpy as np


def calculate2dGaussianArray(xSize, ySize, sigmaX, sigmaY=None, inFrequencySpace=False, fftShift=False):
    """Calculate a 2D Gaussian function either in image or Fourier space.

    The Gaussian is always pixel centered in the way to be considered
    symmetric in discrete Fourier transform (DFT).

    Parameters
    ----------
    xSize, ySize : `int`, greater than 0
        Dimensions of the array to create.
    sigmaX : `float`
        The sigma of the 2D Gaussian along the x axis, in pixels, in image
        space.
    sigmaY : `float`, optional
        The sigma of the 2D Gaussian along the y axis, in pixels, in image
        space. Default ``sigmaX``.
    inFrequencySpace : `bool`, optional
        Deafult False. If True, creates the corresponding Gaussian in Fourier
        space.
    fftShift : `bool`, optional
        Default False. If True, the result array is directly created its
        quadrants shifted so that the Gaussian center is at (0,0).

    Returns
    -------
    R : `numpy.ndarray` of `float`

    Notes
    -----
    The Gaussian is scaled with the $1/(2\\pi \\sigma_x \\sigma_y)$ factor to
    be normed to 1 in image space, but the array is not normed, its sum is
    close to 1 only if the tails are not cut off. In frequency space, the
    scaling always equals to 1.

    Note that for an even size array DFT, the covered frequency range is _not
    symmetric_ due to the 0 frequency (see `numpy.fft.fftfreq`). Hence a pixel
    space input should be asymmetric in values to behave as a "symmetric" input
    from the point of the Fourier transform. This is why the Gaussian function
    is always pixel centered.

    Raises
    ------
    """
    if sigmaY is None:
        sigmaY = sigmaX
    # The left half (LH) should be the smaller for an odd dimension in image
    # space
    xSizeLH = xSize//2
    xSizeRH = xSize - xSizeLH
    ySizeLH = ySize//2
    ySizeRH = ySize - ySizeLH

    pixDist = np.arange(np.maximum(xSizeLH, ySizeLH) + 1, dtype=int)
    # Calculate the function in the positive quarter
    twoPi = 2.*np.pi
    if inFrequencySpace:
        sigmaX = xSize/(twoPi*sigmaX)
        sigmaY = ySize/(twoPi*sigmaY)
    yy, xx = np.meshgrid(-0.5*(pixDist[:xSizeLH + 1]/sigmaX)**2,
                         -0.5*(pixDist[:ySizeLH + 1]/sigmaY)**2,
                         indexing='xy')
    D = np.exp(yy + xx)
    if not inFrequencySpace:
        D /= twoPi*sigmaX*sigmaY
    # Indices in D for the whole array
    xx = np.zeros((ySize, xSize), dtype=int)
    yy = np.zeros((ySize, xSize), dtype=int)
    if fftShift:
        xx[:, :xSizeRH] = pixDist[np.newaxis, :xSizeRH]
        xx[:, xSizeRH:] = pixDist[np.newaxis, xSizeRH:0:-1]
        yy[:ySizeRH, :] = pixDist[:ySizeRH, np.newaxis]
        yy[ySizeRH:, :] = pixDist[ySizeRH:0:-1, np.newaxis]
    else:
        xx[:, xSizeLH:] = pixDist[np.newaxis, :xSizeRH]
        xx[:, :xSizeLH] = pixDist[np.newaxis, xSizeLH:0:-1]
        yy[ySizeLH:, :] = pixDist[:ySizeRH, np.newaxis]
        yy[:ySizeLH, :] = pixDist[ySizeLH:0:-1, np.newaxis]

    return D[yy, xx]
