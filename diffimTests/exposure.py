import numpy as np

from .utils import computeClippedImageStats
from .tasks import doDetection, doForcedPhotometry, doMeasurePsf
from .psf import computeMoments
from .afw import arrayToAfwPsf

class Exposure(object):
    def __init__(self, im, psf=None, var=None, metaData=None):
        self.im = im
        self.psf = psf
        self.var = var
        self.metaData = {} if metaData is None else metaData
        if var is not None:
            self.sig, _, _, _ = computeClippedImageStats(var)
            self.sig = np.sqrt(self.sig)
        else:
            _, self.sig, _, _ = computeClippedImageStats(im)

    def setMetaData(self, key, value):
        self.metaData[key] = value

    def calcSNR(self, flux, skyLimited=False):
        psf = self.psf
        sky = self.sig**2.

        if True:  # Try #1 -- works for Gaussian PSFs
            nPix = np.sum(psf / psf.max()) * 2.  # not sure where the 2 comes from but it works for Gaussian PSFs
            #print nPix, np.pi*2.1*2.1*4  # and it equals pi*r1*r2*4.

        if False:  # Try #2 -- also works for Gaussian PSFs
            xgrid, ygrid = np.meshgrid(np.arange(-psf.shape[0]//2.+1, psf.shape[0]//2.+1),
                                       np.arange(-psf.shape[1]//2.+1, psf.shape[1]//2.+1))
            reffsquared = np.sum((xgrid + ygrid)**2. * psf)
            nPix = np.pi * reffsquared * 2.  # again, why the two? This is equal to the above for Gaussian PSFs
            #print nPix

        if True:  # Try #3 -- same
            moments = computeMoments(psf, p=2.)
            nPix = np.pi * moments[0] * moments[1] * 4.
            #print nPix

        if False:
            # Try #4 -- Surprisingly, the below does exactly the same as the 3 tries above for Gaussian PSFs, but
            # different values for (e.g.) Moffat.
            shape = arrayToAfwPsf(psf).computeShape()
            A = shape.getIxx() + shape.getIyy()
            B = np.sqrt((shape.getIxx() - shape.getIyy())**2. + 4. * shape.getIxy()**2.)
            Rmaj = np.sqrt((A + B) / 2.)
            Rmin = np.sqrt((A - B) / 2.)
            nPix = np.pi * Rmaj * Rmin * 4.  # Note this is the same as shape.getArea() * 4.
            #print nPix

        out = flux / (np.sqrt(flux + nPix * sky))
        if skyLimited:  #  only sky noise matters
            out = flux / (np.sqrt(nPix * sky))
        return out

    def asAfwExposure(self):
        import lsst.afw.image as afwImage
        import lsst.afw.math as afwMath
        import lsst.afw.geom as afwGeom
        import lsst.meas.algorithms as measAlg

        bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Point2I(self.im.shape[0]-1, self.im.shape[1]-1))
        im1ex = afwImage.ExposureF(bbox)
        im1ex.getMaskedImage().getImage().getArray()[:, :] = self.im
        im1ex.getMaskedImage().getVariance().getArray()[:, :] = self.var
        psfShape = self.psf.shape[0]//2
        psfBox = afwGeom.Box2I(afwGeom.Point2I(-psfShape, -psfShape), afwGeom.Point2I(psfShape, psfShape))
        psf = afwImage.ImageD(psfBox)
        psf.getArray()[:, :] = self.psf
        psfK = afwMath.FixedKernel(psf)
        psfNew = measAlg.KernelPsf(psfK)
        im1ex.setPsf(psfNew)
        wcs = makeWcs(naxis1=self.im.shape[0], naxis2=self.im.shape[1])
        im1ex.setWcs(wcs)
        return im1ex

    def doDetection(self, threshold=5.0, doSmooth=True, asDF=False):
        return doDetection(self.asAfwExposure(), threshold=threshold, doSmooth=doSmooth, asDF=asDF)

    def doForcedPhot(self, centroids, transientsOnly=False, asDF=False):
        doForcedPhotometry(centroids, self.asAfwExposure(), transientsOnly=transientsOnly, asDF=asDF)

    def doMeasurePsf(self):
        res = doMeasurePsf(self.asAfwExposure())
        self.psf = afwPsfToArray(res.psf, self.asAfwExposure())  # .computeImage()
        return res


def makeWcs(offset=0, naxis1=1024, naxis2=1153):  # Taken from IP_DIFFIM/tests/testImagePsfMatch.py
    import lsst.daf.base as dafBase
    import lsst.afw.image as afwImage

    metadata = dafBase.PropertySet()
    metadata.set("SIMPLE", "T")
    metadata.set("BITPIX", -32)
    metadata.set("NAXIS", 2)
    metadata.set("NAXIS1", naxis1)
    metadata.set("NAXIS2", naxis2)
    metadata.set("RADECSYS", 'FK5')
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
    return afwImage.makeWcs(metadata)
