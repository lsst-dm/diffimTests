import numpy as np

try:
    import lsst.afw.image as afwImage
    import lsst.afw.math as afwMath
    import lsst.afw.geom as afwGeom
    import lsst.meas.algorithms as measAlg
except:
    pass

from .decorrelation import fixEvenKernel


def alPsfMatchingKernelToArray(psfMatchingKernel, subtractedExposure, coord=None):
    spatialKernel = psfMatchingKernel
    kimg = afwImage.ImageD(spatialKernel.getDimensions())
    bbox = subtractedExposure.getBBox()
    if coord is None:
        xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
        ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
    else:
        xcen, ycen = coord[0], coord[1]
    spatialKernel.computeImage(kimg, True, xcen, ycen)
    return kimg.getArray()


def computeClippedAfwStats(im, numSigmaClip=3., numIter=3, maskIm=None):
    """! Utility function for sigma-clipped array statistics on an image or exposure.
    @param im An afw.Exposure, masked image, or image.
    @return sigma-clipped mean, std, and variance of input array
    """
    statsControl = afwMath.StatisticsControl()
    statsControl.setNumSigmaClip(numSigmaClip)
    statsControl.setNumIter(numIter)
    ignoreMaskPlanes = ["INTRP", "EDGE", "DETECTED", "SAT", "CR", "BAD", "NO_DATA", "DETECTED_NEGATIVE"]
    statsControl.setAndMask(afwImage.MaskU.getPlaneBitMask(ignoreMaskPlanes))
    if maskIm is None:
        statObj = afwMath.makeStatistics(im,
                                         afwMath.MEANCLIP | afwMath.STDEVCLIP | afwMath.VARIANCECLIP,
                                         statsControl)
    else:
        statObj = afwMath.makeStatistics(im, maskIm,
                                         afwMath.MEANCLIP | afwMath.STDEVCLIP | afwMath.VARIANCECLIP,
                                         statsControl)
    mean = statObj.getValue(afwMath.MEANCLIP)
    std = statObj.getValue(afwMath.STDEVCLIP)
    var = statObj.getValue(afwMath.VARIANCECLIP)
    return mean, std, var


def doConvolve(exposure, kernel, use_scipy=False):
    """! Convolve an Exposure with a decorrelation convolution kernel.
    @param exposure Input afw.image.Exposure to be convolved.
    @param kernel Input 2-d numpy.array to convolve the image with
    @param use_scipy Use scipy to do convolution instead of afwMath
    @return a new Exposure with the convolved pixels and the (possibly
    re-centered) kernel.

    @note We use afwMath.convolve() but keep scipy.convolve for debugging.
    @note We re-center the kernel if necessary and return the possibly re-centered kernel
    """
    outExp = kern = None
    fkernel = fixEvenKernel(kernel)
    if use_scipy:
        pci = scipy.ndimage.filters.convolve(exposure.getMaskedImage().getImage().getArray(),
                                             fkernel, mode='constant', cval=np.nan)
        outExp = exposure.clone()
        outExp.getMaskedImage().getImage().getArray()[:, :] = pci
        kern = fkernel

    else:
        kern = arrayToAfwKernel(fkernel)
        outExp = exposure.clone()  # Do this to keep WCS, PSF, masks, etc.
        convCntrl = afwMath.ConvolutionControl(False, True, 0)
        afwMath.convolve(outExp.getMaskedImage(), exposure.getMaskedImage(), kern, convCntrl)

    return outExp, kern


def arrayToAfwKernel(array):
    kernelImg = afwImage.ImageD(array.shape[0], array.shape[1])
    kernelImg.getArray()[:, :] = array
    kern = afwMath.FixedKernel(kernelImg)
    maxloc = np.unravel_index(np.argmax(array), array.shape)
    kern.setCtrX(maxloc[0])
    kern.setCtrY(maxloc[1])
    return kern


def arrayToAfwPsf(array):
    psfcK = arrayToAfwKernel(array)
    psfNew = measAlg.KernelPsf(psfcK)
    return psfNew


def afwPsfToArray(psf, img=None, coord=None):
    if coord is None and img is not None:
        bbox = img.getBBox()
        xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
        ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
    elif coord is not None:
        xcen, ycen = coord[0], coord[1]
    else:
        xcen = ycen = 256.
    out = None
    try:
        out = psf.computeImage(afwGeom.Point2D(xcen, ycen)).getArray()
    except:
        pass
    return out


def afwPsfToShape(psf, img=None, coord=None):
    if coord is None and img is not None:
        bbox = img.getBBox()
        xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
        ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
    elif coord is not None:
        xcen, ycen = coord[0], coord[1]
    else:
        return psf.computeShape()
    out = None
    try:
        out = psf.computeShape(afwGeom.Point2D(xcen, ycen))
    except:
        pass
    return out


# Compute mean of variance plane. Can actually get std of image plane if
# actuallyDoImage=True and statToDo=afwMath.VARIANCECLIP
def computeVarianceMean(exposure, actuallyDoImage=False, statToDo=afwMath.MEANCLIP):
    statsControl = afwMath.StatisticsControl()
    statsControl.setNumSigmaClip(3.)
    statsControl.setNumIter(3)
    ignoreMaskPlanes = ("INTRP", "EDGE", "DETECTED", "SAT", "CR", "BAD", "NO_DATA", "DETECTED_NEGATIVE")
    statsControl.setAndMask(afwImage.MaskU.getPlaneBitMask(ignoreMaskPlanes))
    imToDo = exposure.getMaskedImage().getVariance()
    if actuallyDoImage:
        imToDo = exposure.getMaskedImage().getImage()
    statObj = afwMath.makeStatistics(imToDo, exposure.getMaskedImage().getMask(),
                                     statToDo, statsControl)
    var = statObj.getValue(statToDo)
    return var



# Compute ALZC correction kernel from matching kernel
# Here we use a constant kernel, just compute it for the center of the image.
def performALZCExposureCorrection(templateExposure, exposure, subtractedExposure, psfMatchingKernel, log):
    kimg = alPsfMatchingKernelToArray(psfMatchingKernel, subtractedExposure)
    # Compute the images' sigmas (sqrt of variance)
    sig1 = templateExposure.getMaskedImage().getVariance().getArray()
    sig2 = exposure.getMaskedImage().getVariance().getArray()
    sig1squared, _, _, _ = computeClippedImageStats(sig1)
    sig2squared, _, _, _ = computeClippedImageStats(sig2)
    corrKernel = computeDecorrelationKernel(kimg, sig1squared, sig2squared)
    # Eventually, use afwMath.convolve(), but for now just use scipy.
    log.info("ALZC: Convolving.")
    pci, _ = doConvolve(subtractedExposure.getMaskedImage().getImage().getArray(),
                        corrKernel)
    subtractedExposure.getMaskedImage().getImage().getArray()[:, :] = pci
    log.info("ALZC: Finished with convolution.")

    # Compute the subtracted exposure's updated psf
    psf = afwPsfToArray(subtractedExposure.getPsf(), subtractedExposure)  # .computeImage().getArray()
    psfc = computeCorrectedDiffimPsf(corrKernel, psf, svar=sig1squared, tvar=sig2squared)
    psfcI = afwImage.ImageD(subtractedExposure.getPsf().computeImage().getBBox())
    psfcI.getArray()[:, :] = psfc
    psfcK = afwMath.FixedKernel(psfcI)
    psfNew = measAlg.KernelPsf(psfcK)
    subtractedExposure.setPsf(psfNew)
    return subtractedExposure, corrKernel
