import numpy as np
import scipy
import scipy.stats
import scipy.ndimage.filters
import scipy.signal
import pandas as pd  # We're going to store the results as pandas dataframes.

silent = False
log_level = None
try:
    import lsst.afw.image as afwImage
    import lsst.afw.math as afwMath
    import lsst.afw.geom as afwGeom
    import lsst.meas.algorithms as measAlg
    import lsst.afw.table as afwTable
    import lsst.ip.diffim as ipDiffim
    import lsst.afw.detection as afwDetection
    import lsst.meas.base as measBase
    import lsst.daf.base as dafBase
    import lsst.afw.table.catalogMatches as catMatch
    import lsst.log

    lsst.log.Log.getLogger('afw').setLevel(lsst.log.ERROR)
    lsst.log.Log.getLogger('afw.math').setLevel(lsst.log.ERROR)
    lsst.log.Log.getLogger('afw.image').setLevel(lsst.log.ERROR)
    lsst.log.Log.getLogger('afw.math.convolve').setLevel(lsst.log.ERROR)
    lsst.log.Log.getLogger('TRACE5.afw.math.convolve.convolveWithInterpolation').setLevel(lsst.log.ERROR)
    lsst.log.Log.getLogger('TRACE2.afw.math.convolve.basicConvolve').setLevel(lsst.log.ERROR)
    lsst.log.Log.getLogger('TRACE4.afw.math.convolve.convolveWithBruteForce').setLevel(lsst.log.ERROR)
    log_level = lsst.log.ERROR  # INFO
    import lsst.log.utils as logUtils
    logUtils.traceSetAt('afw', 0)
except Exception as e:
    if not silent:
        print e
        print "LSSTSW has not been set up."


# print 'HERE'

from .utils import *
from .exposure import *
from .diffimTests import *
from .catalog import *
from .plot import *
from . import multi
#from . import afw
#from . import tasks
#from . import psf
#from .makeFakeImages import *
#from . import alardLuptonCustom
#from . import decorrelation
from .imageMapReduce import *
from .imageMapperSubtasks import *
from .zogyTask import *
