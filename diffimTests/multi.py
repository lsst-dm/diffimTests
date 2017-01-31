import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed

from .diffimTests import DiffimTest
from .tasks import doMeasurePsf
from .afw import afwPsfToArray, afwPsfToShape, arrayToAfwPsf
from .psf import computeMoments, resizePsf


def getNumCores():
    num_cores = multiprocessing.cpu_count()
    if num_cores == 32:
        num_cores = 16
    elif num_cores == 8:
        num_cores = 3
    elif num_cores == 4:
        num_cores = 2
    print 'CORES:', num_cores
    return num_cores


def computeNormedPsfRms(psf1, psf2):
    psf1a = psf1.copy() / psf1.max()
    psf2a = psf2.copy() / psf2.max()
    weights = psf1a * psf2a   # instead of squaring either of them separately
    weights /= weights.mean()
    rms1weighted = np.sqrt(((psf1a - psf2a)**2. * weights).mean())
    return rms1weighted


def runTest(flux, seed=66, n_varSources=10, n_sources=500, remeasurePsfs=[False, False],
            returnObj=False, silent=False, **kwargs):
    sky = kwargs.get('sky', 300.)                           # same default as makeFakeImages()
    psf1 = kwargs.get('psf1', [1.6, 1.6])                   # same default as makeFakeImages()
    psf2 = kwargs.get('psf2', [1.8, 2.2])                   # same default as makeFakeImages()
    templateNoNoise = kwargs.get('templateNoNoise', False)  # same default as makeFakeImages()
    skyLimited = kwargs.get('skyLimited', False)            # ditto.
    addPresub = kwargs.get('addPresub', True)  # Probably want this True, but it slows things down

    varFlux2 = flux
    if not hasattr(varFlux2, "__len__"):
        varFlux2 = np.repeat(flux, n_varSources)

    if type(remeasurePsfs[0]) is bool and remeasurePsfs[0]:
        remeasurePsfs[0] = 'psfex'
    if type(remeasurePsfs[1]) is bool and remeasurePsfs[1]:
        remeasurePsfs[1] = 'psfex'

    #print sky, psf1, psf2, varFlux2, n_sources, templateNoNoise, skyLimited, seed, kwargs
    testObj = DiffimTest(varFlux2=varFlux2, #sky=sky, #psf1=psf1, psf2=psf2,
                         n_sources=n_sources, #templateNoNoise=templateNoNoise, skyLimited=skyLimited,
                         seed=seed, **kwargs)

    psf1 = rms1 = shape1 = moments1 = inputShape1 = normedRms1 = inputPsf1 = None
    if type(remeasurePsfs[0]) is not bool or remeasurePsfs[0] != False:
        # re-measure the PSF of the template, save the stats on the orig. and new PSF
        try:
            inputPsf1 = testObj.im1.psf.copy()
            im1 = testObj.im1.asAfwExposure()
            res1 = doMeasurePsf(im1, detectThresh=5.0, measurePsfAlg=remeasurePsfs[0])
            psf1 = afwPsfToArray(res1.psf, im1)
            psf1 = resizePsf(psf1, inputPsf1.shape)
            psf1a = psf1.copy()
            psf1anorm = psf1a[np.abs(psf1a) >= 1e-3].sum()
            psf1a /= psf1anorm

            rms1 = np.sqrt(((psf1a - inputPsf1)**2.).mean())
            normedRms1 = computeNormedPsfRms(psf1a, inputPsf1)
            sh = arrayToAfwPsf(inputPsf1).computeShape()
            inputShape1 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            sh = afwPsfToShape(res1.psf, im1)
            shape1 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            moments1 = computeMoments(psf1)

            psf1b = psf1.copy()
            psf1b[psf1b < 0] = 0
            psf1b[0:10,0:10] = psf1b[31:41,31:41] = 0
            psf1b /= psf1b.sum()
            testObj.im1.psf = psf1b
        except Exception as e:
            print 'HERE1:', e
            psf1 = rms1 = shape1 = moments1 = inputShape1 = normedRms1 = inputPsf1 = None
            #raise e

    psf2 = rms2 = shape2 = moments2 = inputShape2 = normedRms2 = inputPsf2 = None
    if type(remeasurePsfs[1]) is not bool or remeasurePsfs[1] != False:
        # re-measure the PSF of the science image, save the stats on the orig. and new PSF
        try:
            inputPsf2 = testObj.im2.psf.copy()
            im2 = testObj.im2.asAfwExposure()
            res2 = doMeasurePsf(im2, detectThresh=5.0, measurePsfAlg=remeasurePsfs[1])
            psf2 = afwPsfToArray(res2.psf, im2)
            psf2 = resizePsf(psf2, inputPsf2.shape)
            psf2a = psf2.copy()
            psf2anorm = psf2a[np.abs(psf2a) >= 2e-3].sum()
            psf2a /= psf2anorm

            rms2 = np.sqrt(((psf2a - inputPsf2)**2.).mean())
            normedRms2 = computeNormedPsfRms(psf2a, inputPsf2)
            sh = arrayToAfwPsf(inputPsf2).computeShape()
            inputShape2 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            sh = afwPsfToShape(res2.psf, im2)
            shape2 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            moments2 = computeMoments(psf2)

            psf2b = psf2.copy()
            psf2b[psf2b < 0] = 0
            psf2b[0:10,0:10] = psf2b[31:41,31:41] = 0
            psf2b /= psf2b.sum()
            testObj.im2.psf = psf2b
        except Exception as e:
            print 'HERE2:', e
            psf2 = rms2 = shape2 = moments2 = inputShape2 = normedRms2 = inputPsf2 = None
            #raise e

    # This function below is set to *not* plot but it runs `runTest` and outputs the `runTest` results
    # and a dataframe with forced photometry results. So we use this instead of `runTest` directly.
    # Note `addPresub=True` may not always be necessary and will slow it down a bit.
    res = df = sources = None
    try:
        res = testObj.runTest(returnSources=True, **kwargs)
        sources = res['sources']
        df, _ = testObj.doPlotWithDetectionsHighlighted(runTestResult=res, transientsOnly=True,
                                                        addPresub=addPresub, xaxisIsScienceForcedPhot=False,
                                                        actuallyPlot=False, skyLimited=skyLimited)
        if not returnObj:  # delete for space savings
            del res['sources']

    except Exception as e:
        if not silent:
            print 'ERROR RUNNING SEED:', seed #, 'FLUX:', varFlux2
        #raise e

    out = {'result': res, 'flux': flux, 'df': df}
    out['sky'] = sky
    out['n_varSources'] = n_varSources
    out['n_sources'] = n_sources
    out['templateNoNoise'] = templateNoNoise
    out['skyLimited'] = skyLimited
    out['seed'] = seed

    out['templateSNR'] = testObj.im1.calcSNR(flux, skyLimited=skyLimited)
    out['scienceSNR'] = testObj.im2.calcSNR(flux, skyLimited=skyLimited)

    if remeasurePsfs[0] or remeasurePsfs[1]:
        psfout = {'psf1': psf1, 'psf2': psf2,
                  'inputPsf1': inputPsf1, 'inputPsf2': inputPsf2,
                  'rms1': rms1, 'rms2': rms2,
                  'shape1': shape1, 'shape2': shape2,
                  'inputShape1': inputShape1, 'inputShape2': inputShape2,
                  'moments1': moments1, 'moments2': moments2,
                  'nSources': n_sources, 'seed': seed,
                  'normedRms1': normedRms1, 'normedRms2': normedRms2}
        out['psfInfo'] = psfout

    if returnObj:
        out['obj'] = testObj
    return out


# Using flux=620 for SNR=5 (see cell #4 of notebook '30. 4a. other psf models-real PSFs')
def runMultiDiffimTests(varSourceFlux=620., nStaticSources=500, n_runs=100, num_cores=None, **kwargs):
    if not hasattr(varSourceFlux, "__len__"):
        varSourceFlux = [varSourceFlux]
    if not hasattr(nStaticSources, "__len__"):
        nStaticSources = [nStaticSources]
    inputs = [(f, ns, seed) for f in varSourceFlux for ns in nStaticSources
              for seed in np.arange(66, 66+n_runs, 1)]
    print 'RUNNING:', len(inputs)
    if num_cores is None:
        num_cores = getNumCores()
    testResults = Parallel(n_jobs=num_cores, verbose=4)(
        delayed(runTest)(flux=i[0], n_sources=i[1], seed=i[2], **kwargs) for i in inputs)
    return testResults


def plotResults(tr, doRates=False, title='', asHist=False, doPrint=True, actuallyPlot=True):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')

    import seaborn as sns
    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    methods = ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_decorr']
    FN = pd.DataFrame({key: np.array([t['result'][key]['FN'] for t in tr]) for key in methods})
    FP = pd.DataFrame({key: np.array([t['result'][key]['FP'] for t in tr]) for key in methods})
    TP = pd.DataFrame({key: np.array([t['result'][key]['TP'] for t in tr]) for key in methods})
    title_suffix = 's'
    if doRates:
        FN /= (FN + TP)
        FP /= (FN + TP)
        TP /= (FN + TP)
        title_suffix = ' rate'
    if doPrint:
        print 'FN:', '\n', FN.mean()
        print 'FP:', '\n', FP.mean()
        print 'TP:', '\n', TP.mean()

    if not actuallyPlot:
        return TP, FP, FN

    matplotlib.rcParams['figure.figsize'] = (18.0, 6.0)
    fig, axes = plt.subplots(nrows=1, ncols=2)

    if not asHist:
        sns.violinplot(data=TP, cut=True, linewidth=0.3, bw=0.25, scale='width', alpha=0.5, ax=axes[0])
        if TP.shape[0] < 500:
            sns.swarmplot(data=TP, color='black', size=3, alpha=0.3, ax=axes[0])
        sns.boxplot(data=TP, saturation=0.5, boxprops={'facecolor': 'None'},
                    whiskerprops={'linewidth': 0}, showfliers=False, ax=axes[0])
        plt.setp(axes[0], alpha=0.3)
        axes[0].set_ylabel('True positive' + title_suffix)
        axes[0].set_title(title)
        sns.violinplot(data=FP, cut=True, linewidth=0.3, bw=0.5, scale='width', ax=axes[1])
        if FP.shape[0] < 500:
            sns.swarmplot(data=FP, color='black', size=3, alpha=0.3, ax=axes[1])
        sns.boxplot(data=FP, saturation=0.5, boxprops={'facecolor': 'None'},
                    whiskerprops={'linewidth': 0}, showfliers=False, ax=axes[1])
        plt.setp(axes[1], alpha=0.3)
        axes[1].set_ylabel('False positive' + title_suffix)
        axes[1].set_title(title)
    else:
        for t in TP:
            sns.distplot(TP[t], label=t, norm_hist=False, ax=axes[0])
        axes[0].set_xlabel('True positive' + title_suffix)
        axes[0].set_title(title)
        legend = axes[0].legend(loc='upper left', shadow=True)
        for t in FP:
            sns.distplot(FP[t], label=t, norm_hist=False, ax=axes[1])
        axes[1].set_xlabel('False positive' + title_suffix)
        axes[1].set_title(title)
        legend = axes[1].legend(loc='upper left', shadow=True)

    return TP, FP, FN

def plotSnrResults(tr, title='', doPrint=True, snrMax=20):
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    import seaborn as sns
    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    if 'df' not in tr[0]:
        return None
    df = pd.concat([t['df'] for t in tr])
    #df.plot.scatter('scienceSNR', 'ZOGY_SNR')

    matplotlib.rcParams['figure.figsize'] = (18.0, 9.0)

    plt.subplots(2, 2)
    plt.subplot(221)
    good = df.scienceSNR.values < 100.
    scienceSNR = df.scienceSNR.values[good]
    plt.scatter(scienceSNR, df.ALstack_SNR.values[good], label='AL', color='b', alpha=0.2)
    plt.scatter(scienceSNR, df.ALstack_decorr_SNR.values[good], label='AL(decorr)', color='g', alpha=0.2)
    plt.scatter(scienceSNR, df.ZOGY_SNR.values[good], label='ZOGY', color='r', alpha=0.2)
    legend = plt.legend(loc='upper left', shadow=True)
    plt.xlabel('Science image SNR (measured)')
    plt.ylabel('Difference image SNR')
    plt.xlim(0, snrMax)
    plt.title(title)

    if doPrint:
        print title, ':'
        tmp = df.ALstack_SNR.values[~np.isnan(df.ALstack_SNR.values)]
        print 'AL:\t\t', np.median(tmp), '+/-', tmp.std()
        tmp = df.ALstack_decorr_SNR.values[~np.isnan(df.ALstack_decorr_SNR.values)]
        print 'AL(decorr):\t', np.median(tmp), '+/-', tmp.std()
        tmp = df.ZOGY_SNR.values[~np.isnan(df.ZOGY_SNR.values)]
        print 'ZOGY:\t\t', np.median(tmp), '+/-', tmp.std()

    plt.subplot(222)
    sns.distplot(df.ALstack_SNR.values[~np.isnan(df.ALstack_SNR.values)], label='AL', norm_hist=False)
    sns.distplot(df.ALstack_decorr_SNR.values[~np.isnan(df.ALstack_decorr_SNR.values)],
                 label='AL(decorr)', norm_hist=False)
    sns.distplot(df.ZOGY_SNR.values[~np.isnan(df.ZOGY_SNR.values)], label='ZOGY', norm_hist=False)
    sns.distplot(scienceSNR[~np.isnan(scienceSNR)], label='Science img (measured)', norm_hist=False)
    #sns.distplot(df.inputSNR.values, label='Input', norm_hist=False)
    plt.plot(np.repeat(df.inputSNR.values.mean(), 2), np.array([0, 0.4]), label='Input SNR', color='k')
    legend = plt.legend(loc='upper left', shadow=True)
    plt.xlabel('SNR')
    plt.ylabel('Frequency')
    plt.xlim(0, snrMax)
    plt.title(title)
    #df[['ZOGY_SNR', 'ALstack_SNR']].plot.hist(bins=20, alpha=0.5)

    plt.subplot(223)
    df.scienceSNR.values[~good] = np.nan
    g = sns.violinplot(data=df[['ALstack_SNR', 'ALstack_decorr_SNR', 'ZOGY_SNR', 'scienceSNR', 'inputSNR']],
                       linewidth=0.3, bw=0.25, scale='width')
    #sns.swarmplot(data=df[['ALstack_SNR', 'ALstack_decorr_SNR', 'ZOGY_SNR', 'scienceSNR', 'inputSNR']],
    #              color='black', size=0.2, ax=g)
    plt.ylabel('SNR')
    plt.ylim(0, snrMax)
    g.set_xticklabels(g.get_xticklabels(), rotation=60)
    plt.title(title)

    ax = plt.subplot(224)
    df2 = df.groupby('inputFlux').mean()
    df2[['inputSNR', 'ZOGY_detected', 'ALstack_detected',
         'ALstack_decorr_detected']].plot(x='inputSNR', alpha=0.5, lw=5, ax=ax)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Input transient SNR')
    plt.ylabel('Fraction detected')

    return df
