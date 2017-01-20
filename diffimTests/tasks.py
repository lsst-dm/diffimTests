import numpy as np

try:
    import lsst.afw.image as afwImage
    import lsst.afw.math as afwMath
    import lsst.afw.table as afwTable
    import lsst.meas.algorithms as measAlg
    import lsst.ip.diffim as ipDiffim
    import lsst.meas.base as measBase
    import lsst.pipe.tasks.measurePsf as measurePsf
    import lsst.log
    log_level = lsst.log.ERROR  # INFO
except Exception as e:
    print(e)

from .afw import alPsfMatchingKernelToArray, computeVarianceMean
from .decorrelation import computeDecorrelationKernel, computeCorrectedDiffimPsf
from .afw import doConvolve, afwPsfToArray
from .catalog import catalogToDF


def doAlInStack(im1, im2, doWarping=False, doDecorr=True, doPreConv=False,
                spatialBackgroundOrder=0, spatialKernelOrder=0):

    preConvKernel = None
    im2c = im2
    if doPreConv:
        #doDecorr = False  # Right now decorr with pre-conv doesn't work
        preConvKernel = im2.psf
        im2c, kern = doConvolve(im2, preConvKernel, use_scipy=False)

    config = ipDiffim.ImagePsfMatchTask.ConfigClass()
    config.kernel.name = "AL"
    config.selectDetection.thresholdValue = 5.0  # default is 10.0 but this is necessary for very dense fields
    subconfig = config.kernel.active
    config.kernel.active.spatialKernelOrder = spatialBackgroundOrder  # 1  # make 0 since that is set in the default simulation setup.
    config.kernel.active.spatialBgOrder = spatialKernelOrder
    #config.kernel.active.alardMinSig = 0.55  # Default is 0.7 but this is better for my simulations ???
    subconfig.afwBackgroundConfig.useApprox = False
    subconfig.constantVarianceWeighting = False
    subconfig.singleKernelClipping = False
    subconfig.spatialKernelClipping = False
    subconfig.fitForBackground = False

    task = ipDiffim.ImagePsfMatchTask(config=config)
    task.log.setLevel(log_level)
    result = task.subtractExposures(im1, im2c, doWarping=doWarping)
    result.task = task  # for debugging (e.g. task.metadata.get("ALBasisNGauss")

    if doDecorr:
        kimg = alPsfMatchingKernelToArray(result.psfMatchingKernel, im1)
        #return kimg
        if preConvKernel is not None and kimg.shape[0] < preConvKernel.shape[0]:
            # This is likely brittle and may only work if both kernels are odd-shaped.
            #kimg[np.abs(kimg) < 1e-4] = np.sign(kimg)[np.abs(kimg) < 1e-4] * 1e-8
            #kimg -= kimg[0, 0]
            padSize0 = preConvKernel.shape[0]//2 - kimg.shape[0]//2
            padSize1 = preConvKernel.shape[1]//2 - kimg.shape[1]//2
            kimg = np.pad(kimg, ((padSize0, padSize0), (padSize1, padSize1)), mode='constant',
                          constant_values=0)
            #kimg /= kimg.sum()

            #preConvKernel = preConvKernel[padSize0:-padSize0, padSize1:-padSize1]
            #print kimg.shape, preConvKernel.shape

        sig1squared = computeVarianceMean(im1)
        sig2squared = computeVarianceMean(im2)
        pck = computeDecorrelationKernel(kimg, sig1squared, sig2squared,
                                         preConvKernel=preConvKernel, delta=1.)
        #return kimg, preConvKernel, pck
        diffim, _ = doConvolve(result.subtractedExposure, pck, use_scipy=False)
        #diffim.getMaskedImage().getImage().getArray()[:, ] \
        #    /= np.sqrt(im1.metaData['sky'] + im1.metaData['sky'])
        #diffim.getMaskedImage().getVariance().getArray()[:, ] \
        #    /= np.sqrt(im1.metaData['sky'] + im1.metaData['sky'])

        # For some reason, border areas of img and variance planes can become infinite. Fix it.
        img = diffim.getMaskedImage().getImage().getArray()
        img[~np.isfinite(img)] = np.nan
        img = diffim.getMaskedImage().getVariance().getArray()
        img[~np.isfinite(img)] = np.nan
        # TBD: also need to update the mask as it is not (apparently) set correctly.

        psf = afwPsfToArray(result.subtractedExposure.getPsf(), result.subtractedExposure)  # .computeImage().getArray()
        # NOTE! Need to compute the updated PSF including preConvKernel !!! This doesn't do it:
        psfc = computeCorrectedDiffimPsf(kimg, psf, tvar=sig1squared, svar=sig2squared)
        psfcI = afwImage.ImageD(psfc.shape[0], psfc.shape[1])
        psfcI.getArray()[:, :] = psfc
        psfcK = afwMath.FixedKernel(psfcI)
        psfNew = measAlg.KernelPsf(psfcK)
        diffim.setPsf(psfNew)

        result.decorrelatedDiffim = diffim
        result.preConvKernel = preConvKernel
        result.decorrelationKernel = pck
        result.kappaImg = kimg

    return result


def doForcedPhotometry(centroids, exposure, transientsOnly=False, asDF=False):
    expWcs = exposure.getWcs()
    if type(centroids) is afwTable.SourceCatalog:
        sources = centroids
    else:
        sources = centroidsToCatalog(centroids, expWcs, transientsOnly=transientsOnly)
    config = measBase.ForcedMeasurementTask.ConfigClass()
    config.plugins.names = ['base_TransformedCentroid', 'base_PsfFlux']
    config.slots.shape = None
    config.slots.centroid = 'base_TransformedCentroid'
    config.slots.modelFlux = 'base_PsfFlux'
    measurement = measBase.ForcedMeasurementTask(sources.getSchema(), config=config)
    measCat = measurement.generateMeasCat(exposure, sources, expWcs)
    measurement.attachTransformedFootprints(measCat, sources, exposure, expWcs)
    measurement.run(measCat, exposure, sources, expWcs)

    if asDF:
        measCat = catalogToDF(measCat) #pd.DataFrame({col: measCat.columns[col] for col in measCat.schema.getNames()})
    return measCat, sources


def doDetection(exp, threshold=5.0, thresholdType='stdev', thresholdPolarity='both', doSmooth=True,
                doMeasure=True, asDF=False):
    # Modeled from meas_algorithms/tests/testMeasure.py
    schema = afwTable.SourceTable.makeMinimalSchema()
    config = measAlg.SourceDetectionTask.ConfigClass()
    config.thresholdPolarity = thresholdPolarity
    config.reEstimateBackground = False
    config.thresholdValue = threshold
    config.thresholdType = thresholdType
    detectionTask = measAlg.SourceDetectionTask(config=config, schema=schema)
    detectionTask.log.setLevel(log_level)

    # Do measurement too, so we can get x- and y-coord centroids

    config = measBase.SingleFrameMeasurementTask.ConfigClass()
    # Use the minimum set of plugins required.
    config.plugins = ["base_CircularApertureFlux",
                      "base_PixelFlags",
                      "base_SkyCoord",
                      "base_PsfFlux",
                      "base_GaussianCentroid",
                      "base_GaussianFlux",
                      "base_PeakLikelihoodFlux",
                      "base_PeakCentroid",
                      "base_SdssCentroid",
                      "base_SdssShape",
                      "base_NaiveCentroid",
                      #"ip_diffim_NaiveDipoleCentroid",
                      #"ip_diffim_NaiveDipoleFlux",
                      "ip_diffim_PsfDipoleFlux",
                      "ip_diffim_ClassificationDipole",
                      ]
    config.slots.centroid = "base_GaussianCentroid" #"ip_diffim_NaiveDipoleCentroid"
    #config.plugins["base_CircularApertureFlux"].radii = [3.0, 7.0, 15.0, 25.0]
    #config.slots.psfFlux = "base_CircularApertureFlux_7_0" # Use of the PSF flux is hardcoded in secondMomentStarSelector
    config.slots.calibFlux = None
    config.slots.modelFlux = None
    config.slots.instFlux = None
    config.slots.shape = "base_SdssShape"
    config.doReplaceWithNoise = False
    measureTask = measBase.SingleFrameMeasurementTask(schema, config=config)
    measureTask.log.setLevel(log_level)

    table = afwTable.SourceTable.make(schema)
    sources = detectionTask.run(table, exp, doSmooth=doSmooth).sources

    measureTask.measure(exp, sources)

    if asDF:
        sources = catalogToDF(sources) #pd.DataFrame({col: sources.columns[col] for col in sources.schema.getNames()})

    return sources


def doMeasurePsf(exp, measurePsfAlg='psfex', detectThresh=5.0, startSize=0.01, spatialOrder=1):
    # The old (meas_algorithms) SdssCentroid assumed this by default if it
    # wasn't specified; meas_base requires us to be explicit.
    if exp.getPsf() is not None:
        shape = exp.getPsf().computeImage().getDimensions()
    else:
        shape = [21, 21]
    psf = measAlg.DoubleGaussianPsf(shape[0], shape[1], startSize)
    exp.setPsf(psf)

    im = exp.getMaskedImage().getImage()
    im -= np.median(im.getArray())  # why did I do this?  seems to help sometimes.

    sources = doDetection(exp, threshold=detectThresh)
    #print 'N SOURCES:', len(sources)
    config = measurePsf.MeasurePsfConfig()
    schema = afwTable.SourceTable.makeMinimalSchema()

    if measurePsfAlg is 'psfex':
        try:
            import lsst.meas.extensions.psfex.psfexPsfDeterminer
            config.psfDeterminer['psfex'].spatialOrder = spatialOrder  # 2 is default, 0 seems to kill it
            config.psfDeterminer['psfex'].recentroid = True
            config.psfDeterminer['psfex'].sizeCellX = 256  # default is 256
            config.psfDeterminer['psfex'].sizeCellY = 256
            config.psfDeterminer['psfex'].samplingSize = 1  # default is 1
            config.psfDeterminer.name = 'psfex'
        except ImportError as e:
            print "WARNING: Unable to use psfex: %s" % e
            measurePsfAlg = 'pca'

    if measurePsfAlg is 'pca':
        config.psfDeterminer['pca'].sizeCellX = 128
        config.psfDeterminer['pca'].sizeCellY = 128
        config.psfDeterminer['pca'].spatialOrder = 1
        config.psfDeterminer['pca'].nEigenComponents = 3
        #config.psfDeterminer['pca'].tolerance = 1e-1
        #config.starSelector['objectSize'].fluxMin = 500.
        #config.psfDeterminer['pca'].constantWeight = False
        #config.psfDeterminer['pca'].doMaskBlends = False
        config.psfDeterminer.name = "pca"

    psfDeterminer = config.psfDeterminer.apply()
    #print type(psfDeterminer)
    task = measurePsf.MeasurePsfTask(schema=schema, config=config)
    #task.log.setLevel(log_level)
    result = task.run(exp, sources)
    return result
