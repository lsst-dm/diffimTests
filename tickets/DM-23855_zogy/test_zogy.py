#
# LSST Data Management System
# Copyright 2016-2017 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
import unittest

import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.meas.algorithms as measAlg
import lsst.daf.base as dafBase

from lsst.ip.diffim.zogy import ZogyTask, ZogyConfig

try:
    type(verbose)
except NameError:
    verbose = False


def setup_module(module):
    lsst.utils.tests.init()


def singleGaussian2d(x, y, xc, yc, sigma_x=1., sigma_y=1., theta=0., ampl=1.):
    """! Generate a 2-d Gaussian, possibly elongated and rotated, on a grid of pixel
    coordinates given by x,y.
    @param x,y each a 1-d numpy.array containing x- and y- coordinates for independent variables,
    for example `np.arange(-16, 15)`.
    @param xc,yc each a float giving the centroid of the gaussian
    @param sigma_x,sigma_y each a float giving the sigma of the gaussian
    @param theta a float giving the rotation of the gaussian (degrees)
    @param ampl a float giving the amplitude of the gaussian
    @return a 2-d numpy.array containing the normalized 2-d Gaussian

    @Note this can be done in `astropy.modeling` but for now we have it explicitly here.
    """
    theta = (theta/180.) * np.pi
    cos_theta2, sin_theta2 = np.cos(theta)**2., np.sin(theta)**2.
    sigma_x2, sigma_y2 = sigma_x**2., sigma_y**2.
    a = cos_theta2/(2.*sigma_x2) + sin_theta2/(2.*sigma_y2)
    b = -(np.sin(2.*theta))/(4.*sigma_x2) + (np.sin(2.*theta))/(4.*sigma_y2)
    c = sin_theta2/(2.*sigma_x2) + cos_theta2/(2.*sigma_y2)
    xxc, yyc = x-xc, y-yc
    out = np.exp(-(a*(xxc**2.) + 2.*b*xxc*yyc + c*(yyc**2.)))
    out /= out.sum()
    return out


def makeFakeImages(size=(256, 256), svar=0.04, tvar=0.04, psf1=3.3, psf2=2.2, offset=None,
                   psf_yvary_factor=0., varSourceChange=1/50., theta1=0., theta2=0.,
                   n_sources=50, seed=66, verbose=False):
    """Make two exposures: science and template pair with flux sources and random noise.
    In all cases below, index (1) is the science image, and (2) is the template.

    Parameters
    ----------
    size : `tuple` of `int`
        Image pixel size (x,y). Pixel coordinates are set to
        (-size[0]//2:size[0]//2, -size[1]//2:size[1]//2)
    svar, tvar : `float`, optional
        Per pixel variance of the added noise.
    psf1, psf2 : `float`, optional
        std. dev. of (Gaussian) PSFs for the two images in x,y direction. Default is
        [3.3, 3.3] and [2.2, 2.2] for im1 and im2 respectively.
    offset : `float`, optional
        add a constant (pixel) astrometric offset between the two images.
    psf_yvary_factor : `float`, optional
        psf_yvary_factor vary the y-width of the PSF across the x-axis of the science image (zero,
        the default, means no variation)
    varSourceChange : `float`, optional
        varSourceChange add this amount of fractional flux to a single source closest to
        the center of the science image.
    theta1, theta2: `float`, optional
        PSF Gaussian rotation angles in degrees.
    n_sources : `int`, optional
        The number of sources to add to the images. If zero, no sources are
        generated just background noise.
    seed : `int`, optional
        Random number generator seed.
    verbose : `bool`, optional
        Print some actual values.

    Returns
    -------
    im1, im2 : `lsst.afw.image.Exposure`
        The science and template exposures.

    Notes
    -----
    If ``n_sources > 0`` and ``varSourceChange > 0.`` exactly one source,
    that is closest to the center, will have different fluxes in the two
    generated images. The flux on the science image will be higher by
    ``varSourceChange`` fraction.

    Having sources near the edges really messes up the
    fitting (probably because of the convolution). So we make sure no
    sources are near the edge.

    Also it seems that having the variable source with a large
    flux increase also messes up the fitting (seems to lead to
    overfitting -- perhaps to the source itself). This might be fixed by
    adding more constant sources.
    """
    np.random.seed(seed)

    psf1 = [3.3, 3.3] if psf1 is None else psf1
    if not hasattr(psf1, "__len__") and not isinstance(psf1, str):
        psf1 = [psf1, psf1]
    psf2 = [2.2, 2.2] if psf2 is None else psf2
    if not hasattr(psf2, "__len__") and not isinstance(psf2, str):
        psf2 = [psf2, psf2]
    offset = [0., 0.] if offset is None else offset   # astrometric offset (pixels) between the two images
    if verbose:
        print('Science PSF:', psf1, theta1)
        print('Template PSF:', psf2, theta2)
        print(np.sqrt(psf1[0]**2 - psf2[0]**2))
        print('Offset:', offset)
    # Beware that -N//2 != -1*(N//2) for odd numbers
    xim = np.arange(-size[0]//2, size[0]//2, 1)
    yim = np.arange(-size[1]//2, size[1]//2, 1)
    x0im, y0im = np.meshgrid(xim, yim)

    im1 = np.random.normal(scale=np.sqrt(svar), size=x0im.shape)  # variance of science image
    im2 = np.random.normal(scale=np.sqrt(tvar), size=x0im.shape)  # variance of template

    if n_sources > 0:
        fluxes = np.random.uniform(50, 30000, n_sources)
        xposns = np.random.uniform(xim.min()+16, xim.max()-5, n_sources)
        yposns = np.random.uniform(yim.min()+16, yim.max()-5, n_sources)

        # Make the source closest to the center of the image the one that increases in flux
        ind = np.argmin(xposns**2. + yposns**2.)

        # vary the y-width of psf across x-axis of science image (zero means no variation):
        psf1_yvary = psf_yvary_factor * (yim.mean() - yposns) / yim.max()
        if verbose:
            print('PSF y spatial-variation:', psf1_yvary.min(), psf1_yvary.max())

    for i in range(n_sources):
        flux = fluxes[i]
        tmp = flux * singleGaussian2d(x0im, y0im, xposns[i], yposns[i], psf2[0], psf2[1], theta=theta2)
        im2 += tmp
        if i == ind:
            flux += flux * varSourceChange
        tmp = flux * singleGaussian2d(x0im, y0im, xposns[i]+offset[0], yposns[i]+offset[1],
                                      psf1[0], psf1[1]+psf1_yvary[i], theta=theta1)
        im1 += tmp

    im1_psf = singleGaussian2d(x0im, y0im, 0, 0, psf1[0], psf1[1], theta=theta1)
    im2_psf = singleGaussian2d(x0im, y0im, offset[0], offset[1], psf2[0], psf2[1], theta=theta2)

    def makeWcs(offset=0):
        """ Make a fake Wcs

        Parameters
        ----------
        offset : float
          offset the Wcs by this many pixels.
        """
        # taken from $AFW_DIR/tests/testMakeWcs.py
        metadata = dafBase.PropertySet()
        metadata.set("SIMPLE", "T")
        metadata.set("BITPIX", -32)
        metadata.set("NAXIS", 2)
        metadata.set("NAXIS1", 1024)
        metadata.set("NAXIS2", 1153)
        metadata.set("RADESYS", 'FK5')
        metadata.set("EQUINOX", 2000.)
        metadata.setDouble("CRVAL1", 215.604025685476)
        metadata.setDouble("CRVAL2", 53.1595451514076)
        metadata.setDouble("CRPIX1", 1109.99981456774 + offset)
        metadata.setDouble("CRPIX2", 560.018167811613 + offset)
        metadata.set("CTYPE1", 'RA---SIN')
        metadata.set("CTYPE2", 'DEC--SIN')
        metadata.setDouble("CD1_1", 5.10808596133527E-05)
        metadata.setDouble("CD1_2", 1.85579539217196E-07)
        metadata.setDouble("CD2_2", -5.10281493481982E-05)
        metadata.setDouble("CD2_1", -8.27440751733828E-07)
        return afwGeom.makeSkyWcs(metadata)

    def makeExposure(imgArray, psfArray, imgVariance):
        """! Convert an image numpy.array and corresponding PSF numpy.array into an exposure.

        Add the (constant) variance plane equal to `imgVariance`.

        @param imgArray 2-d numpy.array containing the image
        @param psfArray 2-d numpy.array containing the PSF image
        @param imgVariance variance of input image
        @return a new exposure containing the image, PSF and desired variance plane
        """
        # All this code to convert the template image array/psf array into an exposure.
        bbox = geom.Box2I(geom.Point2I(0, 0), geom.Point2I(imgArray.shape[1]-1, imgArray.shape[0]-1))
        im1ex = afwImage.ExposureD(bbox)
        im1ex.getMaskedImage().getImage().getArray()[:, :] = imgArray
        im1ex.getMaskedImage().getVariance().getArray()[:, :] = imgVariance
        psfBox = geom.Box2I(geom.Point2I(-12, -12), geom.Point2I(12, 12))  # a 25x25 pixel psf
        psf = afwImage.ImageD(psfBox)
        psfBox.shift(geom.Extent2I(-1*(-size[0]//2), -1*(-size[1]//2)))
        im1_psf_sub = psfArray[psfBox.getMinY():psfBox.getMaxY()+1, psfBox.getMinX():psfBox.getMaxX()+1]
        psf.getArray()[:, :] = im1_psf_sub
        psfK = afwMath.FixedKernel(psf)
        psfNew = measAlg.KernelPsf(psfK)
        im1ex.setPsf(psfNew)
        wcs = makeWcs()
        im1ex.setWcs(wcs)
        return im1ex

    im1ex = makeExposure(im1, im1_psf, svar)  # Science image
    im2ex = makeExposure(im2, im2_psf, tvar)  # Template

    return im1ex, im2ex


class ZogyTest(lsst.utils.tests.TestCase):
    """A test case for the Zogy task.
    """

    def setUp(self):
        self.psf1_sigma = 3.3  # sigma of psf of science image
        self.psf2_sigma = 2.2  # sigma of psf of template image

        self.statsControl = afwMath.StatisticsControl()
        self.statsControl.setNumSigmaClip(3.)
        self.statsControl.setNumIter(3)
        self.statsControl.setAndMask(afwImage.Mask
                                     .getPlaneBitMask(["INTRP", "EDGE", "SAT", "CR",
                                                       "DETECTED", "BAD",
                                                       "NO_DATA", "DETECTED_NEGATIVE"]))

    def _setUpImages(self, svar=100., tvar=100., varyPsf=0.):
        """Generate a fake aligned template and science image.
        """
        self.svar = svar  # variance of noise in science image
        self.tvar = tvar  # variance of noise in template image

        seed = 1
        self.im1ex, self.im2ex \
            = makeFakeImages(size=(256, 256), svar=self.svar, tvar=self.tvar,
                             psf1=self.psf1_sigma, psf2=self.psf2_sigma,
                             n_sources=0, psf_yvary_factor=varyPsf, varSourceChange=10.,
                             seed=seed, verbose=False)
        # Create an array corresponding to the "expected" subtraction (noise only)
        np.random.seed(seed)
        self.expectedSubtraction = np.random.normal(scale=np.sqrt(svar), size=self.im1ex.getDimensions())
        self.expectedSubtraction -= np.random.normal(scale=np.sqrt(tvar), size=self.im2ex.getDimensions())
        self.expectedVar = np.var(self.expectedSubtraction)
        self.expectedMean = np.mean(self.expectedSubtraction)

    def _computeVarianceMean(self, maskedIm):
        statObj = afwMath.makeStatistics(maskedIm.getVariance(),
                                         maskedIm.getMask(), afwMath.MEANCLIP,
                                         self.statsControl)
        mn = statObj.getValue(afwMath.MEANCLIP)
        return mn

    def _computePixelVariance(self, maskedIm):
        statObj = afwMath.makeStatistics(maskedIm, afwMath.VARIANCECLIP,
                                         self.statsControl)
        var = statObj.getValue(afwMath.VARIANCECLIP)
        return var

    def _computePixelMean(self, maskedIm):
        statObj = afwMath.makeStatistics(maskedIm, afwMath.MEANCLIP,
                                         self.statsControl)
        var = statObj.getValue(afwMath.MEANCLIP)
        return var

    def tearDown(self):
        del self.im1ex
        del self.im2ex

    def testZogyDiffim(self):
        """Compute Zogy diffims.
        """
        self._setUpImages()
        config = ZogyConfig()
        config.scaleByCalibration = False
        task = ZogyTask(config=config)
        D = task.run(self.im1ex, self.im2ex)
        return D


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
