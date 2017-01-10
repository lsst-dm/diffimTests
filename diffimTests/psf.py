import numpy as np
import scipy.stats


def makePsf(psfSize=22, sigma=[1., 1.], theta=0., offset=[0., 0.], x0=None, y0=None,
            type='gaussian'):
    if x0 is None or y0 is None:
        x = np.arange(-psfSize+1, psfSize, 1)
        y = x.copy()
        y0, x0 = np.meshgrid(x, y)
    psf = None
    if type == 'gaussian':
        psf = singleGaussian2d(x0, y0, offset[0], offset[1], sigma[0], sigma[1], theta=theta)
    elif type == 'doubleGaussian':
        psf = doubleGaussian2d(x0, y0, offset[0], offset[1], sigma[0], sigma[1], theta=theta)
    elif type == 'moffat':
        width = (sigma[0] + sigma[1]) / 2. * 2.35482
        psf = moffat2d(x0, y0, offset[0], offset[1], width) # no elongation for this one
    elif type == 'kolmogorov':
        width = (sigma[0] + sigma[1]) / 2. * 2.35482
        psf = kolmogorov2d(width)  # or this one.

    if psf.shape[0] == x0.shape[0] and psf.shape[1] == x0.shape[1]:
        return psf

    # Kolmogorov doesn't listen to my input dimensions, so fix it here.
    if psf.shape[0] > x0.shape[0]:
        pos_max = np.unravel_index(np.argmax(psf), psf.shape)
        psf = psf[(pos_max[0]-x0.shape[0]//2):(pos_max[0]+x0.shape[0]//2+1),
                  (pos_max[1]-x0.shape[1]//2):(pos_max[1]+x0.shape[1]//2+1)]
    elif psf.shape[0] < x0.shape[0]:
        psf = np.pad(psf, (((x0.shape[0]-psf.shape[0])//2, (x0.shape[0]-psf.shape[0])//2),
                           ((x0.shape[1]-psf.shape[1])//2, (x0.shape[1]-psf.shape[1])//2)),
                     mode='constant')

    psfsum = psf.sum()
    if psfsum != 1.0:
        psf /= psfsum
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
    out /= out.sum()
    return out


def kolmogorov2d(fwhm=1.0, offset=[0., 0.]):
    import galsim
    import scipy.ndimage.interpolation
    #gsp = galsim.GSParams(folding_threshold=1.0/512., maximum_fft_size=12288)
    psf = galsim.Kolmogorov(fwhm=fwhm*0.2, flux=1) #, gsparams=gsp)
    im = psf.drawImage(method='real_space', scale=0.2)
    bounds = im.getBounds()
    arr = im.image.array.reshape(bounds.getXMax(), bounds.getXMax())
    if np.abs(np.array(offset)).sum() > 0:
        arr = scipy.ndimage.interpolation.shift(arr, [offset[0], offset[1]])  # spline interpolation for shift
    else:
        # For the PSF, need to offset by -0.5 in each direction, just because.
        arr = scipy.ndimage.interpolation.shift(arr, [-0.5, -0.5])
    arr /= arr.sum()
    return arr


def computeMoments(psf):
    xgrid, ygrid = np.meshgrid(np.arange(0, psf.shape[0]), np.arange(0, psf.shape[1]))
    xmoment = np.average(xgrid, weights=psf)
    ymoment = np.average(ygrid, weights=psf)
    return xmoment, ymoment


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

    cacheName = PSFLIBDIR + os.path.basename(filename).replace('.fits', '_psf.fits')
    if os.path.exists(cacheName) and not forceReMeasure:
        psf = afwDet.Psf.readFits(cacheName)
        source = 'cached'
    else:
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

    if not os.path.exists(cacheName) or forceReMeasure:
        try:
            os.mkdir(PSFLIBDIR)
        except Exception as e:
            pass
        psf.writeFits(cacheName)

    if asArray:
        psf = afwPsfToArray(psf, im)

    return psf, source
