import numpy as np
import scipy.stats
import scipy.ndimage.interpolation

from .afw import afwPsfToArray
from .utils import memoize

try:
    from lsst.afw.detection import Psf
except:
    print "LSSTSW has not been set up."


def makePsf(psfType='gaussian', sigma=[1., 1.], theta=0., offset=[0., 0.], x0=None, y0=None,
            psfSize=21):
    if x0 is None or y0 is None:
        x = np.arange(-psfSize+1, psfSize, 1)
        y = x.copy()
        y0, x0 = np.meshgrid(x, y)
    psf = None
    applyScalingRotation = False
    width = 1.0
    if isinstance(psfType, str):
        if psfType == 'gaussian':
            psf = singleGaussian2d(x0, y0, offset[0], offset[1], sigma[0], sigma[1], theta=theta)
        elif psfType == 'doubleGaussian':
            psf = doubleGaussian2d(x0, y0, offset[0], offset[1], sigma[0], sigma[1], theta=theta)
        elif psfType == 'moffat':
            applyScalingRotation = True
            width = (sigma[0] + sigma[1]) / 2. * 2.35482
            psf = moffat2d(x0, y0, offset[0], offset[1], width)  # no elongation for this one
            width /= 2.35482
        elif psfType == 'kolmogorov':
            applyScalingRotation = True
            width = (sigma[0] + sigma[1]) / 2. * 2.35482
            psf = kolmogorov2d(width, 0.5, 0.5) #, offset[0]+0.5, offset[1]+0.5)  # or this one.
            width /= 2.35482
        else:  # Try to load psf from psfLib, assuming psfType is a filename
            psf, source = loadPsf(psfType, asArray=False)
            if psf is not None:
                applyScalingRotation = True
                psf = afwPsfToArray(psf)

    elif isinstance(psfType, Psf):  # An actual afwDet.Psf.
        applyScalingRotation = True
        psf = afwPsfToArray(psfType)

    elif isinstance(psfType, np.ndarray):  # A numpy array
        applyScalingRotation = True
        psf = psfType

    # Apply differential scaling or rotation for those that don't have it intrinsically
    if applyScalingRotation:
        if sigma[0] != sigma[1]:
            psf = scipy.ndimage.interpolation.zoom(psf, [sigma[0]/width, sigma[1]/width])
            if theta != 0.:
                psf = scipy.ndimage.interpolation.rotate(psf, theta)

    # Kolmogorov doesn't listen to my input dimensions, so fix it here.
    # Also offsets the psf if offset is non-zero
    psf = resizePsf(psf, x0.shape, offset)

    psfmin = psf.min()
    if not np.isclose(psfmin, 0.0):
        psf = psf - psf.min()
    psfsum = psf.sum()
    if not np.isclose(psfsum, 1.0):
        psf = psf / psfsum
    return psf


def gaussian2d(grid, m=None, s=None):
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
    # This is slow, so we use the func. below instead.
    if m is None:
        m = [0., 0.]
    if s is None:
        s = [1., 1.]
    cov = [[s[0], 0], [0, s[1]]]
    var = scipy.stats.multivariate_normal(mean=m, cov=cov)
    return var.pdf(grid)


def singleGaussian2d(x, y, xc, yc, sigma_x=1., sigma_y=1., theta=0., offset=0.):
    theta = (theta/180.) * np.pi
    cos_theta2, sin_theta2 = np.cos(theta)**2., np.sin(theta)**2.
    sigma_x2, sigma_y2 = sigma_x**2., sigma_y**2.
    a = cos_theta2/(2.*sigma_x2) + sin_theta2/(2.*sigma_y2)
    b = -(np.sin(2.*theta))/(4.*sigma_x2) + (np.sin(2.*theta))/(4.*sigma_y2)
    c = sin_theta2/(2.*sigma_x2) + cos_theta2/(2.*sigma_y2)
    xxc, yyc = x-xc, y-yc
    out = np.exp(-(a*(xxc**2.) + 2.*b*xxc*yyc + c*(yyc**2.)))
    if offset != 0.:
        out += offset
    out /= out.sum()
    return out


def doubleGaussian2d(x, y, xc, yc, sigma_x=1., sigma_y=1., theta=0., offset=0.,
                     a=0.7, sigma_x2=None, sigma_y2=None, theta2=None):
    sigma_x2 = sigma_x * 2. if sigma_x2 is None else sigma_x2
    sigma_y2 = sigma_y * 2. if sigma_y2 is None else sigma_y2
    theta2 = theta if theta2 is None else theta2

    g1 = a * singleGaussian2d(x, y, xc, yc, sigma_x, sigma_y, theta, offset)
    g1 += (1-a) * singleGaussian2d(x, y, xc, yc, sigma_x2, sigma_y2, theta2, offset)
    g1 /= g1.sum()
    return g1


def moffat2d(x, y, xc, yc, fwhm=1., alpha=4.765):
    """Two dimensional Moffat.

    Parameters
    ----------
    amplitude : float
        Amplitude of the model.
    x_0, y_0 : float
        x, y position of the maximum of the Moffat model.
    gamma : float
        Core width of the Moffat model.
    alpha : float
        Power index of the Moffat model.
    """

    #fwhm = 2. * gamma * np.sqrt(2**(1./alpha) - 1)
    gamma = fwhm / (2. * np.sqrt(2**(1./alpha) - 1))
    rr_gg = ((x - xc)**2. + (y - yc)**2.) / gamma**2.
    out = (1 + rr_gg)**(-alpha)
    #print gamma, alpha
    out /= out.sum()
    return out


@memoize
def kolmogorov2d(fwhm=1.0, xoffset=0., yoffset=0.):
    import galsim
    #gsp = galsim.GSParams(folding_threshold=1.0/512., maximum_fft_size=12288)
    psf = galsim.Kolmogorov(fwhm=fwhm, flux=1) #, gsparams=gsp)
    im = psf.drawImage(method='real_space', scale=1.0)
    bounds = im.getBounds()
    arr = im.image.array.reshape(bounds.getXMax(), bounds.getXMax())
    if xoffset + yoffset > 0:
        arr = scipy.ndimage.interpolation.shift(arr, [xoffset, yoffset])  # spline interpolation for shift
    #else:
        # For the PSF, need to offset by -0.5 in each direction, just because.
    #    arr = scipy.ndimage.interpolation.shift(arr, [-0.5, -0.5])
    arr /= arr.sum()
    return arr


def computeMoments(psf, p=1.):
    xgrid, ygrid = np.meshgrid(np.arange(-psf.shape[0]//2.+1, psf.shape[0]//2.+1),
                               np.arange(-psf.shape[1]//2.+1, psf.shape[1]//2.+1))
    xmoment = np.average(xgrid**p, weights=psf.T**p)
    ymoment = np.average(ygrid**p, weights=psf.T**p)
    return xmoment, ymoment


def recenterPsf(psf, offset=[0., 0.]):
    xmoment, ymoment = computeMoments(psf)
    if np.abs(xmoment) > 2. or np.abs(ymoment) > 2.:
        return psf
    if not np.isclose(xmoment, offset[0]) or not np.isclose(ymoment, offset[1]):
        psf = scipy.ndimage.interpolation.shift(psf, (offset[0]-xmoment, offset[1]-ymoment))
    return psf


def resizePsf(psf, shape, offset=[0., 0.]):
    changed = False
    if psf.shape[0] > shape[0]:
        changed = True
        pos_max = np.unravel_index(np.argmax(psf), psf.shape)
        psf = psf[(pos_max[0]-shape[0]//2):(pos_max[0]+shape[0]//2+1), :]
    elif psf.shape[0] < shape[0]:
        changed = True
        psf = np.pad(psf, (((shape[0]-psf.shape[0])//2, (shape[0]-psf.shape[0])//2+1), (0, 0)),
                     mode='constant')
        if psf.shape[0] > shape[0]:
            psf = psf[:-1, :]

    if psf.shape[1] > shape[1]:
        changed = True
        pos_max = np.unravel_index(np.argmax(psf), psf.shape)
        psf = psf[:, (pos_max[1]-shape[1]//2):(pos_max[1]+shape[1]//2+1)]
    elif psf.shape[1] < shape[1]:
        changed = True
        psf = np.pad(psf, ((0, 0), ((shape[1]-psf.shape[1])//2, (shape[1]-psf.shape[1])//2+1)),
                     mode='constant')
        if psf.shape[1] > shape[1]:
            psf = psf[:, :-1]

    if changed:
        psf = recenterPsf(psf, offset)
    return psf


@memoize
def loadPsf(filename, asArray=True, forceReMeasure=False):
    import os
    import lsst.afw.image as afwImage
    import lsst.afw.detection as afwDet
    from .afw import afwPsfToShape, afwPsfToArray
    from .tasks import doMeasurePsf

    #afwData = os.getenv('AFWDATA_DIR')
    #filename = afwData + '/CFHT/D4/cal-53535-i-797722_1.fits'

    PSFLIBDIR = './psfLib/'
    source = None
    psf = None

    cacheName = PSFLIBDIR + os.path.basename(filename).replace('.fits', '_psf.fits')
    if os.path.exists(cacheName) and not forceReMeasure:
        psf = afwDet.Psf.readFits(cacheName)
        source = 'cached'
    elif os.path.exists(filename):
        im = afwImage.ExposureF(filename)
        if im.getPsf() is None or forceReMeasure:
            startSize = 0.1
            if im.getPsf() is not None:
                shape = afwPsfToShape(im.getPsf(), im)
                startSize = shape.getDeterminantRadius() * 2.
            res = None
            try:
                res = doMeasurePsf(im, detectThresh=10.0, startSize=startSize, spatialOrder=2)
            except:
                pass
            if res is None or res.psf.computeShape().getIxx() < 1.0:
                try:
                    res = doMeasurePsf(im, detectThresh=10.0, startSize=startSize*10., spatialOrder=1)
                except:
                    pass
            if res is None or res.psf.computeShape().getIxx() < 1.0:
                try:
                    res = doMeasurePsf(im, detectThresh=10.0, startSize=startSize*60., spatialOrder=1)
                except:
                    pass
            if res is not None:
                psf = res.psf
            source = 'measured'
        else:
            psf = im.getPsf()
            source = 'loaded from image'

    if psf is not None and not os.path.exists(cacheName) or forceReMeasure:
        try:
            os.mkdir(PSFLIBDIR)
        except Exception as e:
            pass
        psf.writeFits(cacheName)

    if asArray and psf is not None:
        psf = afwPsfToArray(psf, im)

    return psf, source
