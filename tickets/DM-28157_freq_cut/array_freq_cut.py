import numpy as np


def makeEllipseQuartersMaskArray(shape, yR, xR):
    """Create a boolean array marking True the ellipse region of yR and xR.

    Origins are in the corners as in DFT space.

    Parameters
    ----------
    shape : `tuple` of `int`
        2 element tuple for the resulting array (y,x) dimensions.

    yR, xR : `float` greater than 0.
        Semi-major and minor axes of the ellipse region in array index units.
        yR and xR must satisfy yR + 2 <= shape[0] and xR + 2 <= shape[1].

    Returns
    -------
    resultArray : `numpy.ndarray` of dtype `bool`

    Notes
    -----
    In accordance with the DFT interpretation of the corners, (0, 0) is the
    origin and the other corner points are already one pixel off this origin.
    Pixels equal to the radius are set to True inclusively.
    """

    rSqArray = np.arange(max(shape[0], shape[1]))**2
    # Calculate the equation of the ellipse for one quarter
    R = rSqArray[:shape[0], np.newaxis]/(yR*yR) + rSqArray[np.newaxis, :shape[1]]/(xR*xR)
    R = (R <= 1.)

    iY = int(np.ceil(yR+2))
    iX = int(np.ceil(xR+2))
    Z = np.zeros(shape, dtype=bool)
    Z[:iY, :iX] = R[:iY, :iX]
    Z[-1:-iY:-1, -1:-iX:-1] |= R[:iY-1, :iX-1]
    Z[:iY, -1:-iX:-1] |= R[:iY, :iX-1]
    Z[-1:-iY:-1, :iX] |= R[:iY-1, :iX]
    return Z


def calculateCutFrequencies(wSig1, wSig2, var1, var2, F1=1., F2=1., limit=0.999):
    """Estimate the radius where fc1,fc2 solutions approach their limit values.

    Parameters
    ----------
    wSig1, wSig2: `float`, greater than 0.
        PSF Gaussian width sigmas in image space.

    var1, var2: `float`, greater than 0.
        Zogy model noise variance.

    F1, F2 : `float`, optional
        Photometric scaling. Defaults to 1.

    limit : `float`, optional
        The cut frequency will be set to limit * total integral.

    Returns
    -------
    freq1, freq2 : `float`
        Estimate of the cutting frequencies within the range 0. - 0.8. The
        resolution is 0.008.

    Notes
    -----
    The routine is not protected from numerical under or overflows but
    the result is correct in these cases, too.

    """
    dSq, step = np.linspace(0., 0.808, 102, retstep=True)
    dSq *= dSq
    fourPiSq = 4. * np.pi * np.pi
    var1PerF1Sq = var1/(F1*F1)
    var2PerF2Sq = var2/(F2*F2)
    sqDiffWidth = wSig2*wSig2 - wSig1*wSig1
    if sqDiffWidth > 0.:
        # fc1 -> 0, 1./F1 factor can be omitted from integral estimation
        fc1 = 1./np.sqrt(var1PerF1Sq
                         + var2PerF2Sq * np.exp(fourPiSq*sqDiffWidth*dSq))
        # fc2 -> 1./noiseSigma2
        # Flip and subtract baseline so that fc2 -> 0. too
        fc2 = -1./(F2 * np.sqrt(var2PerF2Sq
                                + var1PerF1Sq * np.exp(-fourPiSq*sqDiffWidth*dSq)))
        fc2 += 1./np.sqrt(var2)
        fc2 = np.fabs(fc2)  # Ensure that the tail does not go below zero
    else:
        sqDiffWidth = -sqDiffWidth
        # fc1 -> 1./noiseSigma1
        fc1 = -1./(F1 * np.sqrt(var1PerF1Sq
                                + var2PerF2Sq * np.exp(-fourPiSq*sqDiffWidth*dSq)))
        fc1 += 1./np.sqrt(var1)
        fc1 = np.fabs(fc1)
        # fc2 -> 0., 1./F2 factor can be omitted from integral estimation
        fc2 = 1./np.sqrt(var2PerF2Sq
                         + var1PerF1Sq * np.exp(fourPiSq*sqDiffWidth*dSq))

    # The 2D integral would be 2*np.pi for radial symmetry which factors out
    # using threshold
    csFc1 = np.cumsum(fc1)
    thresh1 = csFc1[-1] * limit
    freq1 = np.searchsorted(csFc1, thresh1) * step

    csFc2 = np.cumsum(fc2)
    thresh2 = csFc2[-1] * limit
    freq2 = np.searchsorted(csFc2, thresh2) * step

    return freq1, freq2


def calculateGaussianCutFrequency(wSig1, limit=0.999):
    """
    wSig1: sigma in image space
    f1, f2 : frequency cut off values for integrated limit

    Returns
    -------
    freq1, freq2 : Estimate of frequency cut within range 0. - 0.8

    Notes
    -----

    The routine is not protected from numerical under or overflows but
    the result is correct in these cases, too.

    If the total integral is not approached by the frequency 0.8, the result
    may be an underestimation.
    """
    # twoPiSq = 2. * np.pi * np.pi
    wSig = 1./(2.*np.pi*wSig1)
    # print("Freq space sigma:", wSig)
    dSq, step = np.linspace(0., 0.808, 102, retstep=True)
    dSq *= dSq
    fc1 = np.exp(-0.5*dSq/(wSig*wSig))
    # The 2D integral would be 2*np.pi for radial symmetry which factors out
    # using threshold
    csFc1 = np.cumsum(fc1)
    thresh1 = csFc1[-1] * limit
    freq1 = np.searchsorted(csFc1, thresh1) * step

    return freq1


def calculate2dGaussianV2(xSize, ySize, sigmaXSq, sigmaYSq=None,
                          inFrequencySpace=False, fftShift=False):
    """Calculate a 2D Gaussian function either in image or Fourier space.

    The Gaussian is always pixel centered in the way to be considered
    symmetric in discrete Fourier transform (DFT).

    Parameters
    ----------
    xSize, ySize : `int`, greater than 0
        Dimensions of the array to create.
    sigmaXSq : `float`, must not be 0.
        The sigma squared of the 2D Gaussian along the x axis, in pixels, in
        image space. Can be negative to produce a diverging exponential as for
        the case of the ratio of two Gaussians.
    sigmaYSq : `float`, optional
        The sigma of the 2D Gaussian along the y axis, in pixels, in image
        space. Can be negative. Defaults to ``sigmaXSq``.
    inFrequencySpace : `bool`, optional
        Deafult False. If True, creates the corresponding Gaussian in Fourier
        space. In Fourier space the Gaussian is not normed, it corresponds to
        the transform of a normed Gaussian in image space with the given
        widths.
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
    close to 1 only if the tails are not cut off. In frequency space or if
    either sigmaSq is negative, there is no normalization, the scaling factor
    equals to 1.

    Note that for an even size array DFT, the covered frequency range is _not
    symmetric_ due to the 0 frequency (see `numpy.fft.fftfreq`). Hence a pixel
    space input should be asymmetric in values to behave as a "symmetric" input
    from the point of the Fourier transform. This is why the Gaussian function
    is always pixel centered.

    TODO: Add overflow protection for the diverging case.

    Raises
    ------
    """
    if sigmaYSq is None:
        sigmaYSq = sigmaXSq
    # The left half (LH) should be the smaller for an odd dimension in image
    # space
    xSizeLH = xSize//2
    xSizeRH = xSize - xSizeLH
    ySizeLH = ySize//2
    ySizeRH = ySize - ySizeLH

    pixDist = np.arange(np.maximum(xSizeLH, ySizeLH) + 1, dtype=int)
    # Calculate the function in the positive quarter
    twoPi = 2.*np.pi
    fourPiSq = twoPi*twoPi
    if inFrequencySpace:
        sigmaXSq = xSize*xSize/(fourPiSq*sigmaXSq)
        sigmaYSq = ySize*ySize/(fourPiSq*sigmaYSq)
    yy, xx = np.meshgrid(-0.5*(pixDist[:xSizeLH + 1])**2/sigmaXSq,
                         -0.5*(pixDist[:ySizeLH + 1])**2/sigmaYSq,
                         indexing='xy')
    # TODO: add overflow protection
    D = np.exp(yy + xx)
    if not inFrequencySpace:
        D /= twoPi*np.sqrt(sigmaXSq)*np.sqrt(sigmaYSq)
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


def calculateDirectModelFc1Fc2(xSize, ySize, wSig1, wSig2, var1, var2, F1=1., F2=1.):
    """Create an array of the Gaussian model solutions of fc1, fc2 in Fourier
    space.

    Parameters
    ----------
    xSize, ySize : `int`, greater than 0
        Dimensions of the array to create.
    wSig1, wSig2 : `float`, greater than 0.
        PSF Gaussian width sigmas in image space.
    F1, F2 : `float`, optional
        Photometric scaling. Defaults to 1.

    Returns
    -------
    fc1, fc2 : `numpy.ndarray` of `float`
        The calculated Gaussian estimate solutions for the matching kernels
        in Fourier space. Note the arrays are real as the image space
        representation is symmetric.

    Notes
    -----
    The code is not overflow protected.
    The code does not support wSig1==wSig2 at the moment.
    """
    var1PerF1Sq = var1/(F1*F1)
    var2PerF2Sq = var2/(F2*F2)
    sqDiffWidth = wSig2*wSig2 - wSig1*wSig1
    if sqDiffWidth > 0.:
        # fc1 -> 0
        D = calculate2dGaussianV2(xSize, ySize, sigmaXSq=-sqDiffWidth,
                                  inFrequencySpace=True, fftShift=True)
        fc1 = 1./(F1 * np.sqrt(var1PerF1Sq
                               + var2PerF2Sq * D**2))
        # fc2 -> 1./noiseSigma2
        D = calculate2dGaussianV2(xSize, ySize, sigmaXSq=sqDiffWidth,
                                  inFrequencySpace=True, fftShift=True)
        fc2 = 1./(F2 * np.sqrt(var2PerF2Sq
                               + var1PerF1Sq * D**2))
    else:
        sqDiffWidth = -sqDiffWidth
        # fc1 -> 1./noiseSigma1
        D = calculate2dGaussianV2(xSize, ySize, sigmaXSq=sqDiffWidth,
                                  inFrequencySpace=True, fftShift=True)
        fc1 = 1./(F1 * np.sqrt(var1PerF1Sq
                               + var2PerF2Sq * D**2))
        # fc2 -> 0.
        D = calculate2dGaussianV2(xSize, ySize, sigmaXSq=-sqDiffWidth,
                                  inFrequencySpace=True, fftShift=True)
        fc2 = 1./(F2 * np.sqrt(var2PerF2Sq
                               + var1PerF1Sq * D**2))
    return fc1, fc2
