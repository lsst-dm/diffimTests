import numpy as np

import lsst.pex.config as pexConfig
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.pipe.base as pipeBase
import lsst.ip.diffim as ipDiffim
import lsst.log

from .imageMapReduce import (ImageMapReduceConfig, ImageMapperSubtask,
                             ImageMapperSubtaskConfig)
from .utils import computeClippedImageStats
#from . import afw
#from . import decorrelation
from . import zogy

__all__ = ['DecorrelateALKernelMapperSubtask', 'DecorrelateALKernelMapReduceConfig',
           'ZogyMapReduceConfig', 'ZogyMapperSubtask',
           'SpatialDecorrelateALKernelMapperSubtask', 'SpatialDecorrelateALKernelMapReduceConfig']


class DecorrelateALKernelMapperSubtask(ipDiffim.DecorrelateALKernelTask, ImageMapperSubtask):
    """Task to be used as an ImageMapperSubtask for computing
    A&L decorrelation on subimages on a grid across a A&L difference image.

    This task subclasses DecorrelateALKernelTask in order to implement
    all of that task's configuration parameters, as well as its `run` method.
    """
    ConfigClass = ipDiffim.DecorrelateALKernelConfig
    _DefaultName = 'ip_diffim_decorrelateALKernelMapper'

    def __init__(self, *args, **kwargs):
        ipDiffim.DecorrelateALKernelTask.__init__(self, *args, **kwargs)

    def run(self, subExposure, expandedSubExposure, fullBBox,
            template, science, alTaskResult=None, psfMatchingKernel=None,
            preConvKernel=None, **kwargs):
        """Perform decorrelation operation on `subExposure`, using
        `expandedSubExposure` to allow for invalid edge pixels arising from
        convolutions.

        This method performs A&L decorrelation on `subExposure` using
        local measures for image variances and PSF. `subExposure` is a
        sub-exposure of the non-decorrelated A&L diffim. It also
        requires the corresponding sub-exposures of the template
        (`template`) and science (`science`) exposures.

        Parameters
        ----------
        subExposure : afw.Exposure
            the sub-exposure of the diffim
        expandedSubExposure : afw.Exposure
            the expanded sub-exposure upon which to operate
        fullBBox : afwGeom.BoundingBox
            the bounding box of the original exposure
        template : afw.Exposure
            the corresponding sub-exposure of the template exposure
        science : afw.Exposure
            the corresponding sub-exposure of the science exposure
        alTaskResult : pipeBase.Struct
            the result of A&L image differencing on `science` and
            `template`, importantly containing the resulting
            `psfMatchingKernel`. Can be `None`, only if
            `psfMatchingKernel` is not `None`.
        psfMatchingKernel : Alternative parameter for passing the
            A&L `psfMatchingKernel` directly.
        kwargs :
            additional keyword arguments propagated from
            `ImageMapReduceTask.run`.

        Returns
        -------
        A `pipeBase.Struct containing the result of the `subExposure`
        processing, labelled 'subExposure'. It also returns the
        'decorrelationKernel', although that currently is not used.

        Notes
        -----
        This `run` method accepts parameters identical to those of
        `ImageMapperSubtask.run`, since it is called from the
        `ImageMapperTask`.  See that class for more information.
        """
        templateExposure = template  # input template
        scienceExposure = science  # input science image
        if alTaskResult is None and psfMatchingKernel is None:
            raise ValueError('Both alTaskResult and psfMatchingKernel cannot be None')
        psfMatchingKernel = alTaskResult.psfMatchingKernel if alTaskResult is not None else psfMatchingKernel

        # subExp and expandedSubExp are subimages of the (un-decorrelated) diffim!
        # So here we compute corresponding subimages of templateExposure and scienceExposure
        subExp2 = scienceExposure.Factory(scienceExposure, expandedSubExposure.getBBox())
        subExp1 = templateExposure.Factory(templateExposure, expandedSubExposure.getBBox())

        # Prevent too much log INFO verbosity from DecorrelateALKernelTask.run
        logLevel = self.log.getLevel()
        self.log.setLevel(lsst.log.WARN)
        res = ipDiffim.DecorrelateALKernelTask.run(self, subExp2, subExp1, expandedSubExposure,
                                                   psfMatchingKernel)
        self.log.setLevel(logLevel)  # reset the log level

        diffim = res.correctedExposure.Factory(res.correctedExposure, subExposure.getBBox())
        out = pipeBase.Struct(subExposure=diffim, decorrelationKernel=res.correctionKernel)
        return out


class DecorrelateALKernelMapReduceConfig(ImageMapReduceConfig):
    """Configuration parameters for the ImageMapReduceTask to direct it to use
       DecorrelateALKernelMapperSubtask as its mapperSubtask for A&L decorrelation.
    """
    mapperSubtask = pexConfig.ConfigurableField(
        doc='A&L decorrelation subtask to run on each sub-image',
        target=DecorrelateALKernelMapperSubtask
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
        padSize = kwargs.get('padSize', 7)

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
        subExp1 = template.Factory(template, expandedSubExp.getBBox())
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
                                                inImageSpace=imageSpace, padSize=padSize)

            tmpIM.getImage().getArray()[:, :] = D_zogy
            tmpIM.getVariance().getArray()[:, :] = var_zogy

        else:
            S, S_var, Pd, Fd = zogy.computeZogyScorr(subarr1, subarr2, subvar1, subvar2,
                                                     psf1b, psf2b, sig1=sig1, sig2=sig2,
                                                     xVarAst=0., yVarAst=0.,
                                                     inImageSpace=imageSpace, padSize=padSize)

            tmpIM.getImage().getArray()[:, :] = S
            tmpIM.getVariance().getArray()[:, :] = S_var

        # need to eventually compute diffim PSF and set it here.
        out = tmpExp.Factory(tmpExp, subExp.getBBox())

        return pipeBase.Struct(subExposure=out)


class ZogyMapReduceConfig(ImageMapReduceConfig):
    mapperSubtask = pexConfig.ConfigurableField(
        doc='Zogy subtask to run on each sub-image',
        target=ZogyMapperSubtask
    )


## =========================== EXPERIMENT ===================

class SpatialDecorrelateALKernelMapperSubtask(ipDiffim.DecorrelateALKernelTask, ImageMapperSubtask):
    """Task to be used as an ImageMapperSubtask for performing
    A&L decorrelation on subimages on a grid across a A&L difference image.

    THIS IS DESIGNED TO JUST RETURN A CoaddPsf INTENDED FOR CONVOLUTION
    WITH THE DIFFIM, BUT THAT IS NOT POSSIBLE.

    This task subclasses DecorrelateALKernelTask in order to implement
    all of that task's configuration parameters, as well as its `run` method.
    """
    ConfigClass = ipDiffim.DecorrelateALKernelConfig
    _DefaultName = 'ip_diffim_spatialDecorrelateALKernelMapper'

    def __init__(self, *args, **kwargs):
        ipDiffim.DecorrelateALKernelTask.__init__(self, *args, **kwargs)

    def run(self, subExposure, expandedSubExposure, fullBBox,
            template, science, alTaskResult=None, psfMatchingKernel=None,
            preConvKernel=None, returnDiffimPsf=False, **kwargs):
        """Perform decorrelation operation on `subExposure`, using
        `expandedSubExposure` to allow for invalid edge pixels arising from
        convolutions.

        This method performs A&L decorrelation on `subExposure` using
        local measures for image variances and PSF. `subExposure` is a
        sub-exposure of the non-decorrelated A&L diffim. It also
        requires the corresponding sub-exposures of the template
        (`template`) and science (`science`) exposures.

        Parameters
        ----------
        subExposure : afw.Exposure
            the sub-exposure of the diffim
        expandedSubExposure : afw.Exposure
            the expanded sub-exposure upon which to operate
        fullBBox : afwGeom.BoundingBox
            the bounding box of the original exposure
        template : afw.Exposure
            the corresponding sub-exposure of the template exposure
        science : afw.Exposure
            the corresponding sub-exposure of the science exposure
        alTaskResult : pipeBase.Struct
            the result of A&L image differencing on `science` and
            `template`, importantly containing the resulting
            `psfMatchingKernel`. Can be `None`, only if
            `psfMatchingKernel` is not `None`.
        psfMatchingKernel : Alternative parameter for passing the
            A&L `psfMatchingKernel` directly.
        kwargs :
            additional keyword arguments propagated from
            `ImageMapReduceTask.run`.

        Returns
        -------
        A `pipeBase.Struct containing the result of the `subExposure`
        processing, labelled 'subExposure'. It also returns the
        'decorrelationKernel', although that currently is not used.

        Notes
        -----
        This `run` method accepts parameters identical to those of
        `ImageMapperSubtask.run`, since it is called from the
        `ImageMapperTask`.  See that class for more information.
        """
        templateExposure = template  # input template
        scienceExposure = science  # input science image
        if alTaskResult is None and psfMatchingKernel is None:
            raise ValueError('Both alTaskResult and psfMatchingKernel cannot be None')
        psfMatchingKernel = alTaskResult.psfMatchingKernel if alTaskResult is not None else psfMatchingKernel

        # subExp and expandedSubExp are subimages of the (un-decorrelated) diffim!
        # So here we compute corresponding subimages of templateExposure and scienceExposure
        subExp2 = scienceExposure.Factory(scienceExposure, expandedSubExposure.getBBox())
        subExp1 = templateExposure.Factory(templateExposure, expandedSubExposure.getBBox())

        # Prevent too much log INFO verbosity from DecorrelateALKernelTask.run
        #logLevel = self.log.getLevel()
        #self.log.setLevel(lsst.log.WARN)
        #res = ipDiffim.DecorrelateALKernelTask.run(self, subExp2, subExp1, expandedSubExposure,
        #                                           psfMatchingKernel)
        svar = ipDiffim.DecorrelateALKernelTask.computeVarianceMean(self, subExp2)
        tvar = ipDiffim.DecorrelateALKernelTask.computeVarianceMean(self, subExp1)

        kimg = afwImage.ImageD(psfMatchingKernel.getDimensions())
        bbox = subExposure.getBBox()
        xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
        ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
        psfMatchingKernel.computeImage(kimg, True, xcen, ycen)
        kernel = ipDiffim.DecorrelateALKernelTask._computeDecorrelationKernel(kappa=kimg.getArray(),
                                                                            svar=svar, tvar=tvar)

        if not returnDiffimPsf:
            kernelImg = afwImage.ImageD(kernel.shape[0], kernel.shape[1])
            kernelImg.getArray()[:, :] = kernel
            kern = afwMath.FixedKernel(kernelImg)
            maxloc = np.unravel_index(np.argmax(kernel), kernel.shape)
            kern.setCtrX(maxloc[0])
            kern.setCtrY(maxloc[1])

        else:  # Compute the subtracted exposure's updated psf
            psf = subExposure.getPsf().computeImage(afwGeom.Point2D(xcen, ycen)).getArray()
            psfc = ipDiffim.DecorrelateALKernelTask.computeCorrectedDiffimPsf(kernel, psf, svar=svar, tvar=tvar)
            psfcI = afwImage.ImageD(psfc.shape[0], psfc.shape[1])
            psfcI.getArray()[:, :] = psfc
            kern = afwMath.FixedKernel(psfcI)

        psf = measAlg.KernelPsf(kern)
        out = pipeBase.Struct(psf=psf, bbox=subExposure.getBBox())
        return out


class SpatialDecorrelateALKernelMapReduceConfig(ImageMapReduceConfig):
    """Configuration parameters for the ImageMapReduceTask to direct it to use
       SpatialDecorrelateALKernelMapperSubtask as its mapperSubtask for A&L decorrelation.
    """
    mapperSubtask = pexConfig.ConfigurableField(
        doc='A&L decorrelation subtask to run on each sub-image',
        target=SpatialDecorrelateALKernelMapperSubtask
    )

    reduceOperation = pexConfig.ChoiceField(
        dtype=str,
        doc="""Operation to use for reducing subimages into new image.""",
        default="coaddPsf",
        allowed={
            "none": """simply return a list of values and don't re-map results into
                       a new image (noop operation)""",
            "coaddPsf": """Instead of constructing an Exposure, take a list of returned
                       PSFs and use CoaddPsf to construct a single PSF that covers the
                       entire input exposure""",
        }
    )

