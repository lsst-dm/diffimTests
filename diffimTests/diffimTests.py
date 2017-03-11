import numpy as np
import pandas as pd

from .makeFakeImages import makeFakeImages
from .exposure import Exposure
from .zogy import performZogy, performZogyImageSpace, computeZogyDiffimPsf, computeZogy_Scorr
from .tasks import doDetection, doForcedPhotometry
from .catalog import centroidsToCatalog, catalogToDF, computeOffsets
from .utils import computeClippedImageStats
from .plot import plotImageGrid

__all__ = ['DiffimTest']


class DiffimTest(object):
    def __init__(self, doInit=True, **kwargs):
        self.args = kwargs

        if doInit:
            # Generate images and PSF's with the same dimension as the image (used for A&L)
            #im1, im2, P_r, P_n, im1_var, im2_var, self.centroids, \
            #    self.changedCentroidInd = makeFakeImages(**kwargs)
            result = makeFakeImages(**kwargs)
            im1, im2, P_r, P_n = result['im1'], result['im2'], result['im1_psf'], result['im2_psf']
            im1_var, im2_var, self.centroids = result['im1_var'], result['im2_var'], result['centroids']
            self.changedCentroidInd = result['changedCentroidInd']
            self.variablePsf = result['variablePsf']

            self.kwargs = kwargs

            sky = kwargs.get('sky', [300., 300])
            if not hasattr(sky, "__len__"):
                sky = [sky, sky]

            self.im1 = Exposure(im1, P_r, im1_var)
            self.im1.setMetaData('sky', sky[0])

            self.im2 = Exposure(im2, P_n, im2_var)
            self.im2.setMetaData('sky', sky[1])

            self.psf1_orig = self.im1.psf
            self.psf2_orig = self.im2.psf

            self.astrometricOffsets = kwargs.get('offset', [0, 0])
            try:
                dx, dy = self.computeAstrometricOffsets(threshold=2.5)  # dont make this threshold smaller!
                self.astrometricOffsets = [dx, dy]
            except Exception as e:
                print(e)
                #pass

            self.D_AL = self.kappa = self.D_Zogy = self.S_Zogy = self.ALres = None  # self.S_corr_Zogy =

    # Ideally call runTest() first so the images are filled in.
    def doPlot(self, centroidCoord=None, include_Szogy=False, addedImgs=None, **kwargs):
        #fig = plt.figure(1, (12, 12))
        imagesToPlot = [self.im1.im, self.im1.var, self.im2.im, self.im2.var]
        titles = ['Template', 'Template var', 'Science img', 'Science var']
        cx = cy = sz = None
        if centroidCoord is not None:
            cx, cy = centroidCoord[0], centroidCoord[1]
            sz = 25
            if len(centroidCoord) == 3:
                sz = centroidCoord[2]
        if self.D_AL is not None:
            imagesToPlot.append(self.D_AL.im)
            titles.append('A&L')
        if self.D_Zogy is not None:
            titles.append('Zogy')
            imagesToPlot.append(self.D_Zogy.im)
        if self.ALres is not None:
            titles.append('A&L(dec)')
            imagesToPlot.append(self.ALres.decorrelatedDiffim.getMaskedImage().getImage().getArray())
            titles.append('A&L')
            imagesToPlot.append(self.ALres.subtractedExposure.getMaskedImage().getImage().getArray())
        if self.D_Zogy is not None and self.ALres is not None:
            titles.append('A&L(dec) - Zogy')  # Plot difference of diffims
            alIm = self.ALres.decorrelatedDiffim.getMaskedImage().getImage().getArray()
            if centroidCoord is not None:
                alIm = alIm[(cx-sz):(cx+sz), (cy-sz):(cy+sz)]
            stats = computeClippedImageStats(alIm)
            print 'A&L(dec):', stats
            #alIm = (alIm - stats[0]) / stats[1]  # need to renormalize the AL image
            stats = computeClippedImageStats(self.D_Zogy.im)
            print 'Zogy:', stats
            zIm = self.D_Zogy.im
            if centroidCoord is not None:
                zIm = zIm[(cx-sz):(cx+sz), (cy-sz):(cy+sz)]
            #zIm = (zIm - stats[0]) / stats[1]
            print 'A&L(dec) - Zogy:', computeClippedImageStats(alIm - zIm)
            imagesToPlot.append(alIm - zIm)
        if self.ALres is not None:
            titles.append('A&L(dec) - A&L')  # Plot difference of diffims
            alIm = self.ALres.decorrelatedDiffim.getMaskedImage().getImage().getArray()
            zIm = self.ALres.subtractedExposure.getMaskedImage().getImage().getArray()
            print 'A&L(dec) - A&L:', computeClippedImageStats(alIm - zIm)
            imagesToPlot.append(alIm - zIm)
        if include_Szogy and self.S_Zogy is not None:
            titles.append('S(Zogy)')
            imagesToPlot.append(self.S_Zogy.im)
            titles.append('S(Zogy) var')
            imagesToPlot.append(self.S_Zogy.var)
            titles.append('S(Zogy)/S_var > 5')
            imagesToPlot.append((self.S_Zogy.im / self.S_Zogy.var > 5.) * 10.0)
        if addedImgs is not None:
            for i, img in enumerate(addedImgs):
                titles.append('Added ' + str(i))
                print 'Added ' + str(i) + ':', computeClippedImageStats(img)
                imagesToPlot.append(img)

        extent = None
        if centroidCoord is not None:
            for ind, im in enumerate(imagesToPlot):
                if (titles[ind] == 'A&L(dec) - Zogy'): # or (titles[ind] == 'A&L(dec) - A&L'):
                    continue
                imagesToPlot[ind] = im[(cx-sz):(cx+sz), (cy-sz):(cy+sz)]
            extent = ((cx-sz), (cx+sz), (cy-sz), (cy+sz))

        grid = plotImageGrid(imagesToPlot, titles=titles, extent=extent, **kwargs)
        return imagesToPlot, titles, grid

    # Idea is to call test2 = test.clone(), then test2.reverseImages() to then run diffim
    # on im2-im1.
    def reverseImages(self):
        self.im1, self.im2 = self.im2, self.im1
        self.psf1_orig, self.psf2_orig = self.psf2_orig, self.psf1_orig
        self.D_AL = self.kappa = self.D_Zogy = self.S_Zogy = self.ALres = None  # self.S_corr_Zogy = 

    def clone(self):
        out = DiffimTest(imSize=self.im1.im.shape, sky=self.im1.metaData['sky'],
                         doInit=False)
        out.kwargs = self.kwargs
        out.im1, out.im2 = self.im1, self.im2
        out.centroids, out.changedCentroidInd = self.centroids, self.changedCentroidInd
        out.astrometricOffsets = self.astrometricOffsets
        out.ALres = self.ALres
        out.D_AL, out.kappa, out.D_Zogy, out.S_Zogy = self.D_AL, self.kappa, self.D_Zogy, self.S_Zogy
        # out.S_corr_Zogy = self.S_corr_Zogy
        return out

    def doAL(self, spatialKernelOrder=0, spatialBackgroundOrder=1, kernelSize=None, doDecorr=True,
             doPreConv=False, betaGauss=1.):
        if kernelSize is None:
            #if not doPreConv:
            kernelSize = self.im1.psf.shape[0]//2+1  # Hastily assume all PSFs are same sized and square
            #else:
            #    kernelSize = np.floor(self.im1.psf.shape[0] * np.sqrt(2.)).astype(int)//2
            #    if kernelSize % 2 == 0:  # make odd-sized
            #        kernelSize -= 1
        preConvKernel = None
        if doPreConv:
            preConvKernel = self.im2.psf
            if betaGauss == 1.:  # update default, resize the kernel appropriately
                betaGauss = 1./np.sqrt(2.)
        D_AL, D_psf, self.kappa_AL = performAlardLupton(self.im1.im, self.im2.im,
            spatialKernelOrder=spatialKernelOrder,
            spatialBackgroundOrder=spatialBackgroundOrder,
            sig1=self.im1.sig, sig2=self.im2.sig,
            kernelSize=kernelSize, betaGauss=betaGauss,
            doALZCcorrection=doDecorr, im2Psf=self.im2.psf,
            preConvKernel=preConvKernel)
        # This is not entirely correct, we also need to convolve var with the decorrelation kernel (squared):
        var = self.im1.var + scipy.ndimage.filters.convolve(self.im2.var, self.kappa_AL**2., mode='constant',
                                                            cval=np.nan)
        self.D_AL = Exposure(D_AL, D_psf, var)
        #self.D_AL.im /= np.sqrt(self.im1.metaData['sky'] + self.im2.metaData['sky'])  #np.sqrt(var)
        #self.D_AL.var /= np.sqrt(self.im1.metaData['sky'] + self.im2.metaData['sky'])  #np.sqrt(var)
        # TBD: make the returned D an Exposure.
        return self.D_AL, self.kappa_AL

    def computeAstrometricOffsets(self, column='base_GaussianCentroid', fluxCol='base_PsfFlux',
                                  threshold=2.5):
        src1 = self.im1.doDetection(asDF=True)
        src1 = src1[~src1[column + '_flag'] & ~src1[fluxCol + '_flag']]
        src1 = src1[[column + '_x', column + '_y', fluxCol + '_flux']]
        src1.reindex()
        src2 = self.im2.doDetection(asDF=True)
        src2 = src2[~src2[column + '_flag'] & ~src2[fluxCol + '_flag']]
        src2 = src2[[column + '_x', column + '_y', fluxCol + '_flux']]
        src2.reindex()
        dx, dy, _ = computeOffsets(src1, src2, threshold=threshold)
        return dx, dy

    def doZogy(self, computeScorr=True, inImageSpace=False, padSize=15):
        D_Zogy = varZogy = None
        if inImageSpace:
            D_Zogy, varZogy = performZogyImageSpace(self.im1.im, self.im2.im,
                self.im1.var, self.im2.var, self.im1.psf, self.im2.psf,
                sig1=self.im1.sig, sig2=self.im2.sig, padSize=padSize)
        else:  # Do all in fourier space
            D_Zogy, varZogy = performZogy(self.im1.im, self.im2.im,
                self.im1.var, self.im2.var, self.im1.psf, self.im2.psf,
                sig1=self.im1.sig, sig2=self.im2.sig)

        P_D_Zogy = computeZogyDiffimPsf(self.im1.im, self.im2.im,
            self.im1.psf, self.im2.psf, sig1=self.im1.sig, sig2=self.im2.sig,
            Fr=1., Fn=1.)

        D_Zogy[(D_Zogy == 0.) | np.isinf(D_Zogy)] = np.nan
        varZogy[(varZogy == 0.) | np.isnan(D_Zogy) | np.isinf(varZogy)] = np.nan
        self.D_Zogy = Exposure(D_Zogy, P_D_Zogy, varZogy)

        if computeScorr:
            S, S_var, P_D, F_D = computeZogy_Scorr(D_Zogy, self.im1.im, self.im2.im,
                self.im1.var, self.im2.var,
                im1_psf=self.im1.psf, im2_psf=self.im2.psf,
                sig1=self.im1.sig, sig2=self.im2.sig,
                xVarAst=self.astrometricOffsets[0],  # these are already variances.
                yVarAst=self.astrometricOffsets[1],
                padSize=padSize, inImageSpace=inImageSpace)
            self.S_Zogy = Exposure(S, P_D, S_var)

        return self.D_Zogy

    def doAlInStack(self, doWarping=False, doDecorr=True, doPreConv=False,
                    spatialBackgroundOrder=0, spatialKernelOrder=0):
        from .tasks import doAlInStack
        im1 = self.im1.asAfwExposure()
        im2 = self.im2.asAfwExposure()

        result = doAlInStack(im1, im2, doWarping=doWarping, doDecorr=doDecorr, doPreConv=doPreConv,
            spatialBackgroundOrder=spatialBackgroundOrder,
            spatialKernelOrder=spatialKernelOrder)

        return result

    def doReMeasurePsfs(self, whichImages=[1, 2]):
        self.psf1_orig = self.im1.psf
        self.psf2_orig = self.im2.psf
        if 1 in whichImages:
            self.im1.doMeasurePsf(self.im1.asAfwExposure())
        if 2 in whichImages:
            self.im2.doMeasurePsf(self.im2.asAfwExposure())

    def reset(self):
        self.ALres = self.D_Zogy = self.D_AL = self.S_Zogy = None

    # Note I use a dist of sqrt(1.5) because I used to have dist**2 < 1.5.
    def runTest(self, subtractMethods=['ALstack', 'Zogy_S', 'Zogy', 'ALstack_decorr'],
                zogyImageSpace=False, matchDist=np.sqrt(1.5), returnSources=False, **kwargs):
        D_Zogy = S_Zogy = res = D_AL = None
        src = {}
        # Run diffim first
        for subMethod in subtractMethods:
            if subMethod is 'ALstack' or subMethod is 'ALstack_decorr':
                if self.ALres is None:
                    self.ALres = self.doAlInStack(doPreConv=False, doDecorr=True, **kwargs)
            if subMethod is 'Zogy_S':
                if self.S_Zogy is None:
                    self.doZogy(computeScorr=True, inImageSpace=zogyImageSpace)
                S_Zogy = self.S_Zogy
                D_Zogy = self.D_Zogy
            if subMethod is 'Zogy':
                if self.D_Zogy is None:
                    self.doZogy(computeScorr=False, inImageSpace=zogyImageSpace)
                D_Zogy = self.D_Zogy
            if subMethod is 'AL':  # my clean-room (pure python, slow) version of A&L
                try:
                    self.doAL(spatialKernelOrder=0, spatialBackgroundOrder=1)
                    D_AL = self.D_AL
                except Exception as e:
                    print(e)
                    D_AL = None

            # Run detection next
            try:
                if subMethod is 'ALstack':  # Note we DONT set it to 5.5 -- best for noise-free template.
                    src_AL = doDetection(self.ALres.subtractedExposure)
                    src['ALstack'] = src_AL
                elif subMethod is 'ALstack_decorr':
                    src_AL2 = doDetection(self.ALres.decorrelatedDiffim)
                    src['ALstack_decorr'] = src_AL2
                elif subMethod is 'Zogy':
                    src_Zogy = doDetection(D_Zogy.asAfwExposure())
                    src['Zogy'] = src_Zogy
                elif subMethod is 'Zogy_S':  # 'pixel_stdev' doesn't work, so just divide the image by
                    tmp_S = S_Zogy.clone()   # the variance and use a 'value' threshold.
                    tmp_S.im /= tmp_S.var
                    tmp_S.var /= tmp_S.var
                    #src_SZogy = doDetection(S_ZOGY.asAfwExposure(),
                    #                        thresholdType='pixel_stdev', doSmooth=False)
                    src_SZogy = doDetection(tmp_S.asAfwExposure(),
                                            thresholdType='value', doSmooth=False)
                    src['SZogy'] = src_SZogy
                elif subMethod is 'AL' and D_AL is not None:
                    src_AL = doDetection(D_AL.asAfwExposure())
                    src['AL'] = src_AL
            except Exception as e:
                print(e)
                pass

        # Compare detections to input sources and get true positives and false negatives
        changedCentroid = self.getCentroidsCatalog(transientsOnly=True)

        import lsst.afw.table as afwTable
        detections = matchCat = {}
        for key in src:
            srces = src[key]
            srces = srces[~srces['base_PsfFlux_flag']]  # this works!
            matches = afwTable.matchXy(changedCentroid, srces, matchDist)  # these should not need uniquifying
            true_pos = len(matches)
            false_neg = len(changedCentroid) - len(matches)
            false_pos = len(srces) - len(matches)
            detections[key] = {'TP': true_pos, 'FN': false_neg, 'FP': false_pos}

        # sources, fp1, fp2, fp_Zogy, fp_AL, fp_ALd = self.doForcedPhot(transientsOnly=True)
        # if mc_Zogy is not None:
        #     matches = afwTable.matchXy(pp_Zogy, sources, 1.0)
        #     matchedCat = catMatch.matchesToCatalog(matches, metadata)

        if returnSources:
            detections['sources'] = src

        return detections

    def getCentroidsCatalog(self, transientsOnly=False):
        centroids = centroidsToCatalog(self.centroids, self.im1.asAfwExposure().getWcs(),
                                       transientsOnly=transientsOnly)
        return centroids

    def doForcedPhot(self, centroids=None, transientsOnly=False, asDF=False):
        if centroids is None:
            centroids = self.getCentroidsCatalog(transientsOnly=transientsOnly)

        mc1, sources = doForcedPhotometry(centroids, self.im1.asAfwExposure(), asDF=asDF)
        mc2, _ = doForcedPhotometry(centroids, self.im2.asAfwExposure(), asDF=asDF)
        mc_Zogy = mc_AL = mc_ALd = None
        if self.D_Zogy is not None:
            mc_Zogy, _ = doForcedPhotometry(centroids, self.D_Zogy.asAfwExposure(), asDF=asDF)
        if self.ALres is not None:
            mc_AL, _ = doForcedPhotometry(centroids, self.ALres.subtractedExposure, asDF=asDF)
            mc_ALd, _ = doForcedPhotometry(centroids, self.ALres.decorrelatedDiffim, asDF=asDF)

        return sources, mc1, mc2, mc_Zogy, mc_AL, mc_ALd

    # Plot SNRs vs. input fluxes for the diffims and the input images.
    # Derived from notebook '30. 3a. Start from the basics-force phot and matching-restart'.
    # Can just return the dataframe without plotting if desired.
    def doPlotWithDetectionsHighlighted(self, runTestResult=None, transientsOnly=True, addPresub=False,
                                        xaxisIsScienceForcedPhot=False, alpha=0.5,
                                        divideByInput=False, actuallyPlot=True, skyLimited=False,
                                        matchDist=np.sqrt(1.5), **kwargs):

        import lsst.afw.table as afwTable
        import lsst.daf.base as dafBase
        import lsst.afw.table.catalogMatches as catMatch

        if actuallyPlot:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.style.use('ggplot')

        #fp_DIFFIM=fp_Zogy, label='Zogy', color='b', alpha=1.0,

        res = runTestResult
        if runTestResult is None or (runTestResult is not None and 'sources' not in runTestResult):
            res = self.runTest(returnSources=True, matchDist=matchDist)

        src = res['sources']
        #del res['sources']
        #print res

        cats = self.doForcedPhot(transientsOnly=transientsOnly)
        sources, fp1, fp2, fp_Zogy, fp_AL, fp_ALd = cats

        # if xaxisIsScienceForcedPhot is True, then don't use sources['inputFlux_science'] --
        #    use fp2['base_PsfFlux_flux'] instead.
        if not xaxisIsScienceForcedPhot:
            srces = sources['inputFlux_science']
        else:
            srces = fp2['base_PsfFlux_flux']

        df = pd.DataFrame()
        df['inputFlux'] = sources['inputFlux_science']
        df['templateFlux'] = fp1['base_PsfFlux_flux']
        df['scienceFlux'] = fp2['base_PsfFlux_flux']
        df['inputId'] = sources['id']
        df['inputCentroid_x'] = sources['centroid_x']
        df['inputCentroid_y'] = sources['centroid_y']

        snrCalced = self.im2.calcSNR(sources['inputFlux_science'], skyLimited=skyLimited)
        df['inputSNR'] = snrCalced

        fp_DIFFIM = [fp_Zogy, fp_AL, fp_ALd]
        label = ['Zogy', 'ALstack', 'ALstack_decorr']
        color = ['b', 'r', 'g']

        for i, fp_d in enumerate(fp_DIFFIM):
            df[label[i] + '_SNR'] = fp_d['base_PsfFlux_flux']/fp_d['base_PsfFlux_fluxSigma']
            df[label[i] + '_flux'] = fp_d['base_PsfFlux_flux']
            df[label[i] + '_fluxSigma'] = fp_d['base_PsfFlux_fluxSigma']

            if actuallyPlot:
                # Plot all sources
                yvals = fp_d['base_PsfFlux_flux']/fp_d['base_PsfFlux_fluxSigma']
                if divideByInput:
                    yvals /= df['inputSNR']
                plt.scatter(srces, yvals,
                            color=color[i], alpha=alpha, label=label[i])
                #plt.scatter(srces,
                #            fp_d['base_PsfFlux_flux']/fp_d['base_PsfFlux_fluxSigma'],
                #            color='k', marker='x', alpha=alpha, label=None)

            if not xaxisIsScienceForcedPhot:
                matches = afwTable.matchXy(sources, src[label[i]], matchDist)
                metadata = dafBase.PropertyList()
                matchCat = catMatch.matchesToCatalog(matches, metadata)
                sources_detected = catalogToDF(sources)
                detected = np.in1d(sources_detected['id'], matchCat['ref_id'])
                sources_detected = sources_detected[detected]
                sources_detected = sources_detected['inputFlux_science']
                snrCalced_detected = snrCalced[detected]
                fp_Zogy_detected = catalogToDF(fp_d)
                detected = np.in1d(fp_Zogy_detected['id'], matchCat['ref_id'])
                fp_Zogy_detected = fp_Zogy_detected[detected]
            else:
                matches = afwTable.matchXy(fp2, src[label[i]], matchDist)
                metadata = dafBase.PropertyList()
                matchCat = catMatch.matchesToCatalog(matches, metadata)
                sources_detected = catalogToDF(fp2)
                detected = np.in1d(sources_detected['id'], matchCat['ref_id'])
                sources_detected = sources_detected[detected]
                sources_detected = sources_detected['base_PsfFlux_flux']
                snrCalced_detected = snrCalced[detected]
                fp_Zogy_detected = catalogToDF(fp_d)
                detected = np.in1d(fp_Zogy_detected['id'], matchCat['ref_id'])
                fp_Zogy_detected = fp_Zogy_detected[detected]

            df[label[i] + '_detected'] = detected
            if actuallyPlot and len(detected) > 0:
                mStyle = matplotlib.markers.MarkerStyle('o', 'none')
                yvals = fp_Zogy_detected['base_PsfFlux_flux']/fp_Zogy_detected['base_PsfFlux_fluxSigma']
                if divideByInput:
                    yvals /= snrCalced_detected
                plt.scatter(sources_detected, yvals,
                            #label=label[i], s=20, color=color[i], alpha=alpha) #, edgecolors='r')
                            label=None, s=30, edgecolors='r', facecolors='none', marker='o', alpha=1.0) # edgecolors=color[i],

        if addPresub:  # Add measurements in original science and template images
            df['templateSNR'] = fp1['base_PsfFlux_flux']/fp1['base_PsfFlux_fluxSigma']
            df['scienceSNR'] = fp2['base_PsfFlux_flux']/fp2['base_PsfFlux_fluxSigma']
            if actuallyPlot:
                yvals = fp1['base_PsfFlux_flux']/fp1['base_PsfFlux_fluxSigma']
                if divideByInput:
                    yvals /= df['inputSNR']
                plt.scatter(srces, yvals,
                            label='template', color='y', alpha=alpha)
                yvals = fp2['base_PsfFlux_flux']/fp2['base_PsfFlux_fluxSigma']
                if divideByInput:
                    yvals /= df['inputSNR']
                plt.scatter(srces, yvals,
                            label='science', color='orange', alpha=alpha-0.2)

        if actuallyPlot:
            if not divideByInput:
                if xaxisIsScienceForcedPhot:
                    plt.scatter(srces, snrCalced, color='k', alpha=alpha-0.2, s=7, label='Input SNR')
                else:
                    plt.plot(srces, snrCalced, color='k', alpha=alpha-0.2, label='Input SNR')
                plt.ylabel('measured SNR')
            else:
                plt.ylabel('measured SNR / input SNR')

            if len(detected) > 0:
                plt.scatter([10000], [0], s=30, edgecolors='r', facecolors='none', marker='o', label='Detected')
            legend = plt.legend(loc='upper left', scatterpoints=3)
            for label in legend.get_texts():
                label.set_fontsize('x-small')
            if not xaxisIsScienceForcedPhot:
                plt.xlabel('input flux')
            else:
                plt.xlabel('science flux (measured)')

        return df, res
