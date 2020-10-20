import numpy as np
# import lsst.daf.persistence as dafPersist
import lsst.geom as geom
import lsst.pipe.base as pipeBase
import scipy.optimize as scOpt


def getCutoutAndPsf(calexp, srcCat):
    """Get a PSF sized cutout image and the PSF image at the source centroid
    position.


    Parameters
    ----------
    calexp : `lsst.afw.image.exposure.Exposure`
        Exposure to cut from.

    srcCat : `lsst.afw.table.SourceRecord'`
        Source catalog row entry of source to cut.

    Returns
    -------

    """
    psfCenter = geom.Point2D(srcCat['base_SdssCentroid_x'],
                             srcCat['base_SdssCentroid_y'])
    psf = calexp.getPsf()
    # fractional pixels are considered by computeImage()
    # if fraction >=0.5 then the bbox shifts by 1 pixel.
    # The psf center with its bbox in the image will be at the requested
    # fractional pixel position
    psfIm = psf.computeImage(psfCenter)
    bb = psfIm.getBBox()
    R = pipeBase.Struct(cutExp=calexp[bb], psfIm=psfIm, bbox=bb)

    return R


def minimizeDiff(cutExp, psfIm):
    """Perform least squares minimization for one multiplicative factor
    of psfIm to minimise the pixel by pixel difference between the exposure
    and the psf images.

    Parameters
    ----------
    cutExp : `lsst.afw.image.exposure.Exposure`
        The exposure cut out, typically cut by the bbox of the psf.
    psfIm : `lsst.afw.image.Image`
        The computed psf image for the source centroid location.

    Returns
    -------
    R : `lsst.pipe.base.Struct`
        - ``x`` : The optimized multiplicative factor for the psf.
        - ``chi2`` : The difference squared at the value of ``x``.
        - ``diffArr`` : Numpy ndarray of the difference at value ``x``.
    """
    def imgdiff_func(x, im1, im2):
        D = np.ravel(im1 - x[0] * im2)
        # print(np.sum(D**2.0), x)
        return D

    assert cutExp.image.getDimensions() == psfIm.getDimensions()
    shape = cutExp.image.array.shape
    # Must be float64, otherwise initial x step may not make any difference
    # in cost function thus terminates optimisation immediately
    im1 = np.array(cutExp.image.array, dtype=np.float64)
    im2 = np.array(psfIm.array, dtype=np.float64)
    R = scOpt.least_squares(imgdiff_func, 10., x_scale=(1.,),
                            args=(im1, im2))
    assert R.success, "least_squares did not return success"
    # scipy.least_squares uses 0.5 * sum(diff**2) as cost function
    return pipeBase.Struct(x=R.x[0], chi2=2.*R.cost, diffArr=R.fun.reshape(shape))
