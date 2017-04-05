from builtins import range
from builtins import object
#!/usr/bin/env python
#
# LSST Data Management System
#
# Copyright 2008-2016  AURA/LSST.
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
#
import os
import unittest
import numpy

import lsst.utils.tests
import lsst.afw.detection
import lsst.afw.image
import lsst.afw.coord
import lsst.log
import lsst.log.utils
import lsst.meas.modelfit

#   Set trace to 0-5 to view debug messages.  Level 5 enables all traces.
lsst.log.utils.traceSetAt("meas.modelfit.optimizer.Optimizer", -1)
lsst.log.utils.traceSetAt("meas.modelfit.optimizer.solveTrustRegion", -1)


class DoubleShapeletPsfApproxTestMixin(object):

    Algorithm = lsst.meas.modelfit.DoubleShapeletPsfApproxAlgorithm

    def setImage(self, psf):
        if not isinstance(psf, lsst.afw.detection.Psf):
            kernel = lsst.afw.math.FixedKernel(psf)
            psf = lsst.meas.algorithms.KernelPsf(kernel)
        self.psf = psf
        # self.exposure.setPsf(self.psf)

    def initialize(self, psf, ctrl=None, atol=1E-4, **kwds):
        if not isinstance(psf, lsst.afw.detection.Psf):
            kernel = lsst.afw.math.FixedKernel(psf)
            psf = lsst.meas.algorithms.KernelPsf(kernel)
        self.psf = psf
        self.atol = atol
        if ctrl is None:
            ctrl = lsst.meas.modelfit.DoubleShapeletPsfApproxControl()
        self.ctrl = ctrl
        for name, value in kwds.items():
            setattr(self.ctrl, name, value)
        # self.exposure = lsst.afw.image.ExposureF(1, 1)
        # self.exposure.setWcs(
        #     lsst.afw.image.makeWcs(
        #         lsst.afw.coord.IcrsCoord(45*lsst.afw.geom.degrees, 45*lsst.afw.geom.degrees),
        #         lsst.afw.geom.Point2D(0.0, 0.0),
        #         5E-5, 0.0, 0.0, 5E-5
        #     )
        # )
        # self.exposure.setPsf(self.psf)

    def tearDown(self):
        # del self.exposure
        del self.psf
        del self.ctrl
        del self.atol

    def setupTaskConfig(self, config):
        config.slots.shape = None
        config.slots.psfFlux = None
        config.slots.apFlux = None
        config.slots.instFlux = None
        config.slots.modelFlux = None
        config.slots.calibFlux = None
        config.doReplaceWithNoise = False
        config.plugins.names = ["modelfit_DoubleShapeletPsfApprox"]
        config.plugins["modelfit_DoubleShapeletPsfApprox"].readControl(self.ctrl)

    def checkBounds(self, msf):
        """Check that the bounds specified in the control object are met by a MultiShapeletFunction.

        These requirements must be true after a call to any fit method or measure().
        """
        self.assertEqual(len(msf.getComponents()), 2)
        self.assertEqual(
            lsst.shapelet.computeSize(self.ctrl.innerOrder),
            len(msf.getComponents()[0].getCoefficients())
        )
        self.assertEqual(
            lsst.shapelet.computeSize(self.ctrl.outerOrder),
            len(msf.getComponents()[1].getCoefficients())
        )
        self.assertGreater(
            self.ctrl.maxRadiusBoxFraction * (self.psf.computeKernelImage().getBBox().getArea())**0.5,
            lsst.afw.geom.ellipses.Axes(msf.getComponents()[0].getEllipse().getCore()).getA()
        )
        self.assertGreater(
            self.ctrl.maxRadiusBoxFraction * (self.psf.computeKernelImage().getBBox().getArea())**0.5,
            lsst.afw.geom.ellipses.Axes(msf.getComponents()[1].getEllipse().getCore()).getA()
        )
        self.assertLess(
            self.ctrl.minRadius,
            lsst.afw.geom.ellipses.Axes(msf.getComponents()[0].getEllipse().getCore()).getB()
        )
        self.assertLess(
            self.ctrl.minRadius,
            lsst.afw.geom.ellipses.Axes(msf.getComponents()[1].getEllipse().getCore()).getB()
        )
        self.assertLess(
            self.ctrl.minRadiusDiff,
            (msf.getComponents()[1].getEllipse().getCore().getDeterminantRadius()
             - msf.getComponents()[0].getEllipse().getCore().getDeterminantRadius())
        )

    def checkRatios(self, msf):
        """Check that the ratios specified in the control object are met by a MultiShapeletFunction.

        These requirements must be true after initializeResult and fitMoments, but will are relaxed
        in later stages of the fit.
        """
        inner = msf.getComponents()[0]
        outer = msf.getComponents()[1]
        position = msf.getComponents()[0].getEllipse().getCenter()
        self.assertFloatsAlmostEqual(position.getX(), msf.getComponents()[1].getEllipse().getCenter().getX())
        self.assertFloatsAlmostEqual(position.getY(), msf.getComponents()[1].getEllipse().getCenter().getY())
        self.assertFloatsAlmostEqual(outer.evaluate()(position),
                                     inner.evaluate()(position)*self.ctrl.peakRatio)
        self.assertFloatsAlmostEqual(
            outer.getEllipse().getCore().getDeterminantRadius(),
            inner.getEllipse().getCore().getDeterminantRadius() * self.ctrl.radiusRatio
        )

    def makeImages(self, msf):
        """Return an Image of the data and an Image of the model for comparison.
        """
        #dataImage = self.exposure.getPsf().computeKernelImage()
        dataImage = self.psf.computeKernelImage()
        modelImage = dataImage.Factory(dataImage.getBBox())
        msf.evaluate().addToImage(modelImage)
        return dataImage, modelImage

    def checkFitQuality(self, msf):
        """Check the quality of the fit by comparing to the PSF image.
        """
        dataImage, modelImage = self.makeImages(msf)
        self.assertFloatsAlmostEqual(dataImage.getArray(), modelImage.getArray(), atol=self.atol,
                                     plotOnFailure=True)

    def testSingleFramePlugin(self):
        """Run the algorithm as a single-frame plugin and check the quality of the fit.
        """
        config = lsst.meas.base.SingleFrameMeasurementTask.ConfigClass()
        self.setupTaskConfig(config)
        config.slots.centroid = "centroid"
        schema = lsst.afw.table.SourceTable.makeMinimalSchema()
        centroidKey = lsst.afw.table.Point2DKey.addFields(schema, "centroid", "centroid", "pixel")
        task = lsst.meas.base.SingleFrameMeasurementTask(config=config, schema=schema)
        measCat = lsst.afw.table.SourceCatalog(schema)
        measRecord = measCat.addNew()
        measRecord.set(centroidKey, lsst.afw.geom.Point2D(0.0, 0.0))
        task.run(measCat, self.exposure)
        #self.assertFalse(measRecord.get("modelfit_DoubleShapeletPsfApprox_flag"))
        key = lsst.shapelet.MultiShapeletFunctionKey(schema["modelfit"]["DoubleShapeletPsfApprox"])
        msf = measRecord.get(key)
        return msf
        self.checkBounds(msf)
        self.checkFitQuality(msf)

    def testForcedPlugin(self):
        """Run the algorithm as a forced plugin and check the quality of the fit.
        """
        config = lsst.meas.base.ForcedMeasurementTask.ConfigClass()
        self.setupTaskConfig(config)
        config.slots.centroid = "base_TransformedCentroid"
        config.plugins.names |= ["base_TransformedCentroid"]
        refSchema = lsst.afw.table.SourceTable.makeMinimalSchema()
        refCentroidKey = lsst.afw.table.Point2DKey.addFields(refSchema, "centroid", "centroid", "pixel")
        refSchema.getAliasMap().set("slot_Centroid", "centroid")
        refCat = lsst.afw.table.SourceCatalog(refSchema)
        refRecord = refCat.addNew()
        refRecord.set(refCentroidKey, lsst.afw.geom.Point2D(0.0, 0.0))
        refWcs = self.exposure.getWcs()  # same as measurement Wcs
        task = lsst.meas.base.ForcedMeasurementTask(config=config, refSchema=refSchema)
        measCat = task.generateMeasCat(self.exposure, refCat, refWcs)
        task.run(measCat, self.exposure, refCat, refWcs)
        measRecord = measCat[0]
        #self.assertFalse(measRecord.get("modelfit_DoubleShapeletPsfApprox_flag"))
        measSchema = measCat.schema
        key = lsst.shapelet.MultiShapeletFunctionKey(measSchema["modelfit"]["DoubleShapeletPsfApprox"])
        msf = measRecord.get(key)
        return msf
        self.checkBounds(msf)
        self.checkFitQuality(msf)

    def testInitializeResult(self):
        """Test that initializeResult() returns a unit-flux, unit-circle MultiShapeletFunction
        with the right peakRatio and radiusRatio.
        """
        msf = self.Algorithm.initializeResult(self.ctrl)
        return msf
        self.assertFloatsAlmostEqual(msf.evaluate().integrate(), 1.0)
        moments = msf.evaluate().computeMoments()
        axes = lsst.afw.geom.ellipses.Axes(moments.getCore())
        self.assertFloatsAlmostEqual(moments.getCenter().getX(), 0.0)
        self.assertFloatsAlmostEqual(moments.getCenter().getY(), 0.0)
        self.assertFloatsAlmostEqual(axes.getA(), 1.0)
        self.assertFloatsAlmostEqual(axes.getB(), 1.0)
        self.assertEqual(len(msf.getComponents()), 2)
        self.checkRatios(msf)

    def testFitMoments(self):
        """Test that fitMoments() preserves peakRatio and radiusRatio while setting moments
        correctly.
        """
        MOMENTS_RTOL = 1E-13
        image = self.psf.computeKernelImage()
        array = image.getArray()
        bbox = image.getBBox()
        x, y = numpy.meshgrid(
            numpy.arange(bbox.getBeginX(), bbox.getEndX()),
            numpy.arange(bbox.getBeginY(), bbox.getEndY())
        )
        msf = self.Algorithm.initializeResult(self.ctrl)
        return msf
        self.Algorithm.fitMoments(msf, self.ctrl, image)
        self.assertFloatsAlmostEqual(msf.evaluate().integrate(), array.sum(), rtol=MOMENTS_RTOL)
        moments = msf.evaluate().computeMoments()
        q = lsst.afw.geom.ellipses.Quadrupole(moments.getCore())
        cx = (x*array).sum()/array.sum()
        cy = (y*array).sum()/array.sum()
        self.assertFloatsAlmostEqual(moments.getCenter().getX(), cx, rtol=MOMENTS_RTOL)
        self.assertFloatsAlmostEqual(moments.getCenter().getY(), cy, rtol=MOMENTS_RTOL)
        self.assertFloatsAlmostEqual(q.getIxx(), ((x - cx)**2 * array).sum()/array.sum(), rtol=MOMENTS_RTOL)
        self.assertFloatsAlmostEqual(q.getIyy(), ((y - cy)**2 * array).sum()/array.sum(), rtol=MOMENTS_RTOL)
        self.assertFloatsAlmostEqual(q.getIxy(), ((x - cx)*(y - cy)*array).sum()/array.sum(),
                                     rtol=MOMENTS_RTOL)
        self.assertEqual(len(msf.getComponents()), 2)
        self.checkRatios(msf)
        self.checkBounds(msf)

    def testObjective(self):
        """Test that model evaluation agrees with derivative evaluation in the objective object.
        """
        image = self.psf.computeKernelImage()
        msf = self.Algorithm.initializeResult(self.ctrl)
        self.Algorithm.fitMoments(msf, self.ctrl, image)
        moments = msf.evaluate().computeMoments()
        r0 = moments.getCore().getDeterminantRadius()
        objective = self.Algorithm.makeObjective(moments, self.ctrl, image)
        image, model = self.makeImages(msf)
        parameters = numpy.zeros(4, dtype=float)
        parameters[0] = msf.getComponents()[0].getCoefficients()[0]
        parameters[1] = msf.getComponents()[1].getCoefficients()[0]
        parameters[2] = msf.getComponents()[0].getEllipse().getCore().getDeterminantRadius() / r0
        parameters[3] = msf.getComponents()[1].getEllipse().getCore().getDeterminantRadius() / r0
        residuals = numpy.zeros(image.getArray().size, dtype=float)
        objective.computeResiduals(parameters, residuals)
        return msf
        self.assertFloatsAlmostEqual(
            residuals.reshape(image.getHeight(), image.getWidth()),
            image.getArray() - model.getArray()
        )
        step = 1E-6
        derivatives = numpy.zeros((parameters.size, residuals.size), dtype=float).transpose()
        objective.differentiateResiduals(parameters, derivatives)
        for i in range(parameters.size):
            original = parameters[i]
            r1 = numpy.zeros(residuals.size, dtype=float)
            r2 = numpy.zeros(residuals.size, dtype=float)
            parameters[i] = original + step
            objective.computeResiduals(parameters, r1)
            parameters[i] = original - step
            objective.computeResiduals(parameters, r2)
            parameters[i] = original
            d = (r1 - r2)/(2.0*step)
            self.assertFloatsAlmostEqual(
                d.reshape(image.getHeight(), image.getWidth()),
                derivatives[:, i].reshape(image.getHeight(), image.getWidth()),
                atol=1E-11
            )

    def testFitProfile(self):
        """Test that fitProfile() does not modify the ellipticity, that it improves the fit, and
        that small perturbations to the zeroth-order amplitudes and radii do not improve the fit.
        """
        image = self.psf.computeKernelImage()
        msf = self.Algorithm.initializeResult(self.ctrl)
        self.Algorithm.fitMoments(msf, self.ctrl, image)
        prev = lsst.shapelet.MultiShapeletFunction(msf)
        self.Algorithm.fitProfile(msf, self.ctrl, image)
        return msf

        def getEllipticity(m, c):
            s = lsst.afw.geom.ellipses.SeparableDistortionDeterminantRadius(
                m.getComponents()[c].getEllipse().getCore()
            )
            return numpy.array([s.getE1(), s.getE2()])
        self.assertFloatsAlmostEqual(getEllipticity(prev, 0), getEllipticity(msf, 0), rtol=1E-13)
        self.assertFloatsAlmostEqual(getEllipticity(prev, 1), getEllipticity(msf, 1), rtol=1E-13)

        def computeChiSq(m):
            data, model = self.makeImages(m)
            return numpy.sum((data.getArray() - model.getArray())**2)
        bestChiSq = computeChiSq(msf)
        self.assertLessEqual(bestChiSq, computeChiSq(prev))
        step = 1E-4
        for component in msf.getComponents():
            # 0th-order amplitude perturbation
            original = component.getCoefficients()[0]
            component.getCoefficients()[0] = original + step
            self.assertLessEqual(bestChiSq, computeChiSq(msf))
            component.getCoefficients()[0] = original - step
            self.assertLessEqual(bestChiSq, computeChiSq(msf))
            component.getCoefficients()[0] = original
            # Radius perturbation
            original = component.getEllipse()
            component.getEllipse().getCore().scale(1.0 + step)
            self.assertLessEqual(bestChiSq, computeChiSq(msf))
            component.setEllipse(original)
            component.getEllipse().getCore().scale(1.0 - step)
            self.assertLessEqual(bestChiSq, computeChiSq(msf))
            component.setEllipse(original)
        return msf

    def testFitShapelets(self):
        """Test that fitShapelets() does not modify the zeroth order coefficients or ellipse,
        that it improves the fit, and that small perturbations to the higher-order coefficients
        do not improve the fit.
        """
        image = self.psf.computeKernelImage()
        msf = self.Algorithm.initializeResult(self.ctrl)
        self.Algorithm.fitMoments(msf, self.ctrl, image)
        self.Algorithm.fitProfile(msf, self.ctrl, image)
        prev = lsst.shapelet.MultiShapeletFunction(msf)
        self.Algorithm.fitShapelets(msf, self.ctrl, image)
        return msf

        self.assertFloatsAlmostEqual(
            prev.getComponents()[0].getEllipse().getParameterVector(),
            msf.getComponents()[0].getEllipse().getParameterVector()
        )
        self.assertFloatsAlmostEqual(
            prev.getComponents()[1].getEllipse().getParameterVector(),
            msf.getComponents()[1].getEllipse().getParameterVector()
        )

        def computeChiSq(m):
            data, model = self.makeImages(m)
            return numpy.sum((data.getArray() - model.getArray())**2)
        bestChiSq = computeChiSq(msf)
        self.assertLessEqual(bestChiSq, computeChiSq(prev))
        step = 1E-4
        for component in msf.getComponents():
            for i in range(1, len(component.getCoefficients())):
                original = component.getCoefficients()[i]
                component.getCoefficients()[i] = original + step
                self.assertLessEqual(bestChiSq, computeChiSq(msf))
                component.getCoefficients()[i] = original - step
                self.assertLessEqual(bestChiSq, computeChiSq(msf))
                component.getCoefficients()[i] = original
        return msf


class SingleGaussianTestCase(DoubleShapeletPsfApproxTestMixin, lsst.utils.tests.TestCase):

    def setUp(self):
        numpy.random.seed(500)
        DoubleShapeletPsfApproxTestMixin.initialize(
            self, psf=lsst.afw.detection.GaussianPsf(25, 25, 2.0),
            innerOrder=0, outerOrder=0, peakRatio=0.0
        )


class HigherOrderTestCase0(DoubleShapeletPsfApproxTestMixin, lsst.utils.tests.TestCase):

    def setUp(self):
        numpy.random.seed(500)
        image = lsst.afw.image.ImageD(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   "data", "psfs/great3-0.fits"))
        DoubleShapeletPsfApproxTestMixin.initialize(
            self, psf=image,
            innerOrder=3, outerOrder=2,
            atol=0.0005
        )


class HigherOrderTestCase1(DoubleShapeletPsfApproxTestMixin, lsst.utils.tests.TestCase):

    def setUp(self):
        numpy.random.seed(500)
        image = lsst.afw.image.ImageD(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   "data", "psfs/great3-1.fits"))
        DoubleShapeletPsfApproxTestMixin.initialize(
            self, psf=image,
            innerOrder=2, outerOrder=1,
            atol=0.002
        )


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
