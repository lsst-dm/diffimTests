import numpy as np

import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase

from .imageMapReduce import (ImageMapReduceConfig, ImageMapperSubtask,
                             ImageMapperSubtaskConfig)
from .utils import computeClippedImageStats
from . import afw
from . import decorrelation
from . import zogy

__all__ = ['ALdecMapReduceConfig', 'ALdecMapperSubtask',
           'ZogyMapReduceConfig', 'ZogyMapperSubtask']


class ALdecMapperSubtask(ImageMapperSubtask):
    ConfigClass = ImageMapperSubtaskConfig
    _DefaultName = 'diffimTests_ALdecMapperSubtask'

    def __init__(self, *args, **kwargs):
        ImageMapperSubtask.__init__(self, *args, **kwargs)

    def run(self, subExp, expandedSubExp, fullBBox, **kwargs):
        bbox = subExp.getBBox()
        center = ((bbox.getBeginX() + bbox.getEndX()) // 2., (bbox.getBeginY() + bbox.getEndY()) // 2.)
        center = afwGeom.Point2D(center[0], center[1])

        variablePsf2 = kwargs.get('variablePsf', None)
        sigmaSquared = kwargs.get('sigmaSquared', None)
        alTaskResult = kwargs.get('alTaskResult', None)
        im1 = kwargs.get('template', None)  # im1 = input template
        im2 = kwargs.get('science', None)   # im2 = input science image
        preConvKernel = kwargs.get('preConvKernel', None)

        # subExp and expandedSubExp are subimages of the (un-decorrelated) diffim!
        # So here we compute corresponding subimages of im1, and im2
        subExp2 = afwImage.ExposureF(im2, expandedSubExp.getBBox())
        subim2 = subExp2.getMaskedImage()  # expandedSubExp.getMaskedImage()
        if sigmaSquared is None:
            subvar2 = subim2.getVariance().getArray()
            sig2squared = computeClippedImageStats(subvar2).mean
        else:
            sig2squared = sigmaSquared[1]  # for testing, can use the input sigma (global value for entire exposure)

        # Psf and image for template img (index 1)
        subExp1 = afwImage.ExposureF(im1, expandedSubExp.getBBox())
        subim1 = subExp1.getMaskedImage()
        if sigmaSquared is None:
            subvar1 = subim1.getVariance().getArray()
            sig1squared = computeClippedImageStats(subvar1).mean
        else:
            sig1squared = sigmaSquared[0]

        # This code taken more-or-less directly from tasks.doALdecorrelation:
        kimg = afw.alPsfMatchingKernelToArray(alTaskResult.psfMatchingKernel, coord=center)
        dck = decorrelation.computeDecorrelationKernel(kimg, sig1squared, sig2squared,
                                                       preConvKernel=preConvKernel, delta=0.)
        diffim, _ = afw.doConvolve(expandedSubExp, dck, use_scipy=False)
        img = diffim.getMaskedImage().getImage().getArray()
        img[~np.isfinite(img)] = np.nan
        img = diffim.getMaskedImage().getVariance().getArray()
        img[~np.isfinite(img)] = np.nan

        if variablePsf2 is None:
            # psf = afw.afwPsfToArray(alTaskResult.subtractedExposure.getPsf(), coord=center)
            psf = afw.afwPsfToArray(im2.getPsf(), coord=center)
            if psf.shape[0] < psf.shape[1]:  # sometimes CoaddPsf does this.
                psf = np.pad(psf, ((1, 1), (0, 0)), mode='constant')
            elif psf.shape[0] > psf.shape[1]:
                psf = np.pad(psf, ((0, 0), (1, 1)), mode='constant')
            # psf = afw.afwPsfToArray(exposure.getPsf(), coord=center)
        else:
            psf = variablePsf2.getImage(center.getX(), center.getY())

        # NOTE! Need to compute the updated PSF including preConvKernel !!! This doesn't do it:
        psfc = decorrelation.computeCorrectedDiffimPsf(kimg, psf, tvar=sig1squared, svar=sig2squared)
        psfcI = afwImage.ImageD(psfc.shape[0], psfc.shape[1])
        psfcI.getArray()[:, :] = psfc
        psfcK = afwMath.FixedKernel(psfcI)
        psfNew = measAlg.KernelPsf(psfcK)
        out = afwImage.ExposureF(diffim, subExp.getBBox())
        out.setPsf(psfNew)

        return pipeBase.Struct(subExposure=out, decorrelationKernel=dck, psf=psfNew)


class ALdecMapReduceConfig(ImageMapReduceConfig):
    mapperSubtask = pexConfig.ConfigurableField(
        doc='A&L decorrelation subtask to run on each sub-image',
        target=ALdecMapperSubtask
    )


class ZogyMapperSubtask(ImageMapperSubtask):
    ConfigClass = ImageMapperSubtaskConfig
    _DefaultName = 'diffimTests_ZogyMapperSubtask'

    def __init__(self, *args, **kwargs):
        ImageMapperSubtask.__init__(self, *args, **kwargs)

    def run(self, subExp, expandedSubExp, fullBBox, **kwargs):
        bbox = subExp.getBBox()
        center = ((bbox.getBeginX() + bbox.getEndX()) // 2., (bbox.getBeginY() + bbox.getEndY()) // 2.)
        center = afwGeom.Point2D(center[0], center[1])

        variablePsf2 = kwargs.get('variablePsf', None)
        sigmas = kwargs.get('sigmas', None)
        imageSpace = kwargs.get('inImageSpace', False)
        doScorr = kwargs.get('Scorr', False)

        # Psf and image for science img (index 2)
        subExp2 = subExp
        subim2 = expandedSubExp.getMaskedImage()
        subarr2 = subim2.getImage().getArray()
        subvar2 = subim2.getVariance().getArray()
        if sigmas is None:
            sig2 = np.sqrt(computeClippedImageStats(subvar2).mean)
        else:
            sig2 = sigmas[1]  # for testing, can use the input sigma (global value for entire exposure)

        # Psf and image for template img (index 1)
        template = kwargs.get('template')
        subExp1 = afwImage.ExposureF(template, expandedSubExp.getBBox())
        subim1 = subExp1.getMaskedImage()
        subarr1 = subim1.getImage().getArray()
        subvar1 = subim1.getVariance().getArray()
        if sigmas is None:
            sig1 = np.sqrt(computeClippedImageStats(subvar1).mean)
        else:
            sig1 = sigmas[0]

        if variablePsf2 is None:
            psf2 = subExp2.getPsf().computeImage(center).getArray()
        else:
            psf2 = variablePsf2.getImage(center.getX(), center.getY())
        if psf2.shape[0] < psf2.shape[1]:  # sometimes CoaddPsf does this.
            psf2 = np.pad(psf2, ((1, 1), (0, 0)), mode='constant')
        elif psf2.shape[0] > psf2.shape[1]:
            psf2 = np.pad(psf2, ((0, 0), (1, 1)), mode='constant')

        psf1 = template.getPsf().computeImage(center).getArray()
        if psf1.shape[0] < psf1.shape[1]:  # sometimes CoaddPsf does this.
            psf1 = np.pad(psf1, ((1, 1), (0, 0)), mode='constant')
        elif psf1.shape[0] > psf1.shape[1]:
            psf1 = np.pad(psf1, ((0, 0), (1, 1)), mode='constant')

        psf1b = psf1; psf2b = psf2
        if True and psf1.shape[0] == 41:   # it's a measured psf (hack!) Note this *really* helps for measured psfs.
            psf1b = psf1.copy()
            psf1b[psf1b < 0] = 0
            psf1b[0:10, :] = psf1b[:, 0:10] = psf1b[31:41, :] = psf1b[:, 31:41] = 0
            psf1b /= psf1b.sum()

            psf2b = psf2.copy()
            psf2b[psf2b < 0] = 0
            psf2b[0:10, :] = psf2b[:, 0:10] = psf2b[31:41, :] = psf2b[:, 31:41] = 0
            psf2b /= psf2b.sum()

        # from diffimTests.diffimTests ...
        if subarr1.shape[0] < psf1.shape[0] or subarr1.shape[1] < psf1.shape[1]:
            return pipeBase.Struct(subExposure=subExp)

        tmpExp = expandedSubExp.clone()
        tmpIM = tmpExp.getMaskedImage()

        if not doScorr:
            D_zogy, var_zogy = zogy.computeZogy(subarr1, subarr2, subvar1, subvar2,
                                                psf1b, psf2b, sig1=sig1, sig2=sig2,
                                                inImageSpace=imageSpace)

            tmpIM.getImage().getArray()[:, :] = D_zogy
            tmpIM.getVariance().getArray()[:, :] = var_zogy

        else:
            S, S_var, Pd, Fd = zogy.computeZogyScorr(subarr1, subarr2, subvar1, subvar2,
                                                     psf1b, psf2b, sig1=sig1, sig2=sig2,
                                                     xVarAst=0., yVarAst=0.,
                                                     inImageSpace=imageSpace, padSize=7)

            tmpIM.getImage().getArray()[:, :] = S
            tmpIM.getVariance().getArray()[:, :] = S_var

        # need to eventually compute diffim PSF and set it here.
        out = afwImage.ExposureF(tmpExp, subExp.getBBox())

        return pipeBase.Struct(subExposure=out)


class ZogyMapReduceConfig(ImageMapReduceConfig):
    mapperSubtask = pexConfig.ConfigurableField(
        doc='Zogy subtask to run on each sub-image',
        target=ZogyMapperSubtask
    )
