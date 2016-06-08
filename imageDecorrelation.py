import numpy as np
from scipy.stats import sigmaclip
from scipy.fftpack import fft2, ifft2, ifftshift
from scipy.ndimage.filters import convolve

import lsst.afw.image as afwImage
import lsst.meas.algorithms as measAlg
import lsst.afw.math as afwMath

__all__ = ("performExposureDecorrelation")


def computeClippedImageStats(im, low=3, high=3):
    _, low, upp = sigmaclip(im, low=low, high=high)
    tmp = im[(im > low) & (im < upp)]
    mean1 = np.nanmean(tmp)
    sig1 = np.nanstd(tmp)
    return mean1, sig1


def getImageGrid(im):
    xim = np.arange(np.int(-np.floor(im.shape[0]/2.)), np.int(np.floor(im.shape[0]/2)))
    yim = np.arange(np.int(-np.floor(im.shape[1]/2.)), np.int(np.floor(im.shape[1]/2)))
    x0im, y0im = np.meshgrid(xim, yim)
    return x0im, y0im


# Compute the "L(ZOGY)" post-conv. kernel from kfit
# Note unlike previous notebooks, here because the PSF is varying,
# we'll just use `fit2` rather than `im2-conv_im1` as the diffim,
# since `fit2` already incorporates the spatially varying PSF.
# sig1 and sig2 are the same as those input to makeFakeImages().
def computeDecorrelationCorrectionKernel(kappa, sig1=0.2, sig2=0.2):
    def post_conv_kernel_ft2(kernel, sig1=1., sig2=1.):
        kft = fft2(kernel)
        return np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))

    def post_conv_kernel2(kernel, sig1=1., sig2=1.):
        kft = post_conv_kernel_ft2(kernel, sig1, sig2)
        out = ifft2(kft)
        return out

    pck = post_conv_kernel2(kappa, sig1=sig2, sig2=sig1)
    pck = ifftshift(pck.real)

    # I think we may need to "reverse" the PSF, as in the ZOGY (and Kaiser) papers... let's try it.
    # This is the same as taking the complex conjugate in Fourier space before FFT-ing back to real space.
    if False:  # TBD: figure this out.
        pck = pck[::-1, :]

    return pck


# Compute the (corrected) diffim's new PSF
# post_conv_psf = phi_1(k) * sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
# we'll parameterize phi_1(k) as a gaussian with sigma "psfsig1".
# im2_psf is the the psf of im2
def computeCorrectedDiffimPsf(kappa, im2_psf, sig1=0.2, sig2=0.2):
    def post_conv_psf_ft2(psf, kernel, sig1=1., sig2=1.):
        # Pad psf or kernel symmetrically to make them the same size!
        if psf.shape[0] < kernel.shape[0]:
            while psf.shape[0] < kernel.shape[0]:
                psf = np.pad(psf, (1, 1), mode='constant')
        elif psf.shape[0] > kernel.shape[0]:
            while psf.shape[0] > kernel.shape[0]:
                kernel = np.pad(kernel, (1, 1), mode='constant')
        psf_ft = fft2(psf)
        kft = fft2(kernel)
        out = psf_ft * np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))
        return out

    def post_conv_psf(psf, kernel, sig1=1., sig2=1.):
        kft = post_conv_psf_ft2(psf, kernel, sig1, sig2)
        out = ifft2(kft)
        return out

    im2_psf_small = im2_psf
    # First compute the science image's (im2's) psf, subset on -16:15 coords
    if im2_psf.shape[0] > 50:
        x0im, y0im = getImageGrid(im2_psf)
        x = np.arange(-16, 16, 1)
        y = x.copy()
        x0, y0 = np.meshgrid(x, y)
        im2_psf_small = im2_psf[(x0im.max()+x.min()+1):(x0im.max()-x.min()+1),
                                (y0im.max()+y.min()+1):(y0im.max()-y.min()+1)]
    pcf = post_conv_psf(psf=im2_psf_small, kernel=kappa, sig1=sig2, sig2=sig1)
    pcf = pcf.real / pcf.real.sum()
    return pcf


# Compute decorrelation correction kernel from matching kernel
# Here we use a constant kernel, just compute it for the center of the image.
def performExposureDecorrelation(templateExposure, exposure, subtractedExposure, psfMatchingKernel, log):
    spatialKernel = psfMatchingKernel
    kimg = afwImage.ImageD(spatialKernel.getDimensions())
    bbox = subtractedExposure.getBBox()
    xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
    ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
    spatialKernel.computeImage(kimg, True, xcen, ycen)
    # Compute the images' sigmas (sqrt of variance)
    sig1 = templateExposure.getMaskedImage().getVariance().getArray()
    sig2 = exposure.getMaskedImage().getVariance().getArray()
    sig1squared, _ = computeClippedImageStats(sig1)
    sig2squared, _ = computeClippedImageStats(sig2)
    sig1 = np.sqrt(sig1squared)
    sig2 = np.sqrt(sig2squared)
    corrKernel = computeDecorrelationCorrectionKernel(kimg.getArray(), sig1=sig1, sig2=sig2)
    # Eventually, use afwMath.convolve(), but for now we just use scipy.
    log.info("Decorrelation: Convolving.")
    pci = convolve(subtractedExposure.getMaskedImage().getImage().getArray(),
                   corrKernel, mode='constant')
    subtractedExposure.getMaskedImage().getImage().getArray()[:, :] = pci
    log.info("Decorrelation: Finished with convolution.")

    # Compute the subtracted exposure's updated psf
    psf = subtractedExposure.getPsf().computeImage().getArray()
    psfc = computeCorrectedDiffimPsf(corrKernel, psf, sig1=sig1, sig2=sig2)
    psfcI = afwImage.ImageD(subtractedExposure.getPsf().computeImage().getBBox())
    psfcI.getArray()[:, :] = psfc
    psfcK = afwMath.FixedKernel(psfcI)
    psfNew = measAlg.KernelPsf(psfcK)
    subtractedExposure.setPsf(psfNew)
    return subtractedExposure, corrKernel
