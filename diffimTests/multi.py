import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed

from .diffimTests import DiffimTest
from .tasks import measurePsf
from .afw import afwPsfToArray, afwPsfToShape, arrayToAfwPsf
from .psf import computeMoments


def getNumCores():
    num_cores = multiprocessing.cpu_count()
    if num_cores == 32:
        num_cores = 16
    elif num_cores == 8:
        num_cores = 3
    elif num_cores == 4:
        num_cores = 2
    print 'CORES:', num_cores


def computeNormedPsfRms(psf1, psf2):
    psf1a = psf1.copy() / psf1.max()
    psf2a = psf2.copy() / psf2.max()
    weights = psf1a * psf2a   # instead of squaring either of them separately
    weights /= weights.mean()
    rms1weighted = np.sqrt(((psf1a - psf2a)**2. * weights).mean())
    return rms1weighted


def runTest(flux, seed=66, n_varSources=50, n_sources=500, remeasurePsfs=[False, False], **kwargs):
    sky = kwargs.get('sky', 300.)
    psf1 = kwargs.get('psf1', [1.6, 1.6])
    psf2 = kwargs.get('psf2', [1.8, 2.2])
    templateNoNoise = kwargs.get('templateNoNoise', True)
    skyLimited = kwargs.get('skyLimited', True)
    addPresub = kwargs.get('addPresub', True)

    testObj = DiffimTest(sky=sky, psf1=psf1, psf2=psf2, varFlux2=np.repeat(flux, n_varSources),
                         n_sources=n_sources, sourceFluxRange=(200, 20000),
                         templateNoNoise=templateNoNoise, skyLimited=skyLimited, avoidAllOverlaps=15.,
                         seed=seed)

    if remeasurePsfs[0]:  # re-measure the PSF of the template, save the stats on the orig. and new PSF
        psf1 = rms1 = shape1 = moments1 = inputShape1 = normedRms1 = None
        try:
            actualPsf1 = testObj.im1.psf.copy()
            im1 = testObj.im1.asAfwExposure()
            res1 = measurePsf(im1, detectThresh=5.0, measurePsfAlg='psfex')
            psf1 = afwPsfToArray(res1.psf, im1)
            psf1a = psf1.copy()
            psf1anorm = psf1a[np.abs(psf1a) >= 1e-3].sum()
            psf1a /= psf1anorm
            rms1 = np.sqrt(((psf1a - actualPsf1)**2.).mean())
            normedRms1 = computeNormedPsfRms(psf1a, actualPsf1)
            sh = arrayToAfwPsf(actualPsf1).computeShape()
            inputShape1 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            sh = afwPsfToShape(res1.psf, im1)
            shape1 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            moments1 = computeMoments(psf1)
        except Exception as e:
            psf1 = rms1 = shape1 = moments1 = inputShape1 = normedRms1 = None

    if remeasurePsfs[1]:  # re-measure the PSF of the science image, save the stats on the orig. and new PSF
        psf2 = rms2 = shape2 = moments2 = inputShape2 = normedRms2 = None
        try:
            actualPsf2 = testObj.im2.psf.copy()
            im2 = testObj.im2.asAfwExposure()
            res2 = measurePsf(im2, detectThresh=5.0, measurePsfAlg='psfex')
            psf2 = afwPsfToArray(res2.psf, im2)
            psf2a = psf2.copy()
            psf2anorm = psf2a[np.abs(psf2a) >= 2e-3].sum()
            psf2a /= psf2anorm
            rms2 = np.sqrt(((psf2a - actualPsf2)**2.).mean())
            normedRms2 = computeNormedPsfRms(psf2a, actualPsf2)
            sh = arrayToAfwPsf(actualPsf2).computeShape()
            inputShape2 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            sh = afwPsfToShape(res2.psf, im2)
            shape2 = [sh.getDeterminantRadius(), sh.getIxx(), sh.getIyy(), sh.getIxy()]
            moments2 = computeMoments(psf2)
        except Exception as e:
            psf2 = rms2 = shape2 = moments2 = inputShape2 = normedRms2 = None



    # This function below is set to *not* plot but it runs `runTest` and outputs the `runTest` results
    # and a dataframe with forced photometry results. So we use this instead of `runTest` directly.
    # Note `addPresub=True` may not always be necessary and will slow it down a bit.
    df = None
    try:
        res, df = testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=addPresub,
                                                          xaxisIsScienceForcedPhot=False, actuallyPlot=False,
                                                          skyLimited=skyLimited)
    except Exception as e:
        print 'ERROR IN SEED:', seed
        raise e
        res = testObj.runTest(returnSources=True)  # Getting exceptions when no matches, so do this instead.
        src = res['sources']
        del res['sources']

    res['flux'] = flux
    res['df'] = df

    if remeasurePsfs[0] or remeasurePsfs[1]:
        out = {'psf1': psf1, 'psf2': psf2,
               'inputPsf1': actualPsf1, 'inputPsf2': actualPsf2,
               'rms1': rms1, 'rms2': rms2,
               'shape1': shape1, 'shape2': shape2,
               'inputShape1': inputShape1, 'inputShape2': inputShape2,
               'moments1': moments1, 'moments2': moments2,
               'nSources': n_sources, 'seed': seed,
               'normedRms1': normedRms1, 'normedRms2': normedRms2}

        for key in out.keys():
            res[key] = out[key]

    return res


# Using flux=620 for SNR=5 (see cell #4 of notebook '30. 4a. other psf models-real PSFs')
def runMultiDiffimTests(varSourceFlux=620., n_runs=100, num_cores=None, **kwargs):
    inputs = [(f, seed) for f in [varSourceFlux] for seed in np.arange(66, 66+n_runs, 1)]
    print 'RUNNING:', len(inputs)
    if num_cores is None:
        num_cores = getNumCores()
    testResults = Parallel(n_jobs=num_cores, verbose=2)(delayed(runTest)(flux=i[0], seed=i[1], **kwargs)
                                                        for i in inputs)
    return testResults


def plotResults(tr, doRates=False, title='', asHist=False, doPrint=True):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')

    import seaborn as sns
    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    methods = ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_decorr']
    FN = pd.DataFrame({key: np.array([t[key]['FN'] for t in tr]) for key in methods})
    FP = pd.DataFrame({key: np.array([t[key]['FP'] for t in tr]) for key in methods})
    TP = pd.DataFrame({key: np.array([t[key]['TP'] for t in tr]) for key in methods})
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

    matplotlib.rcParams['figure.figsize'] = (18.0, 6.0)
    fig, axes = plt.subplots(nrows=1, ncols=2)

    if not asHist:
        sns.violinplot(data=TP, cut=True, linewidth=0.3, bw=0.25, ax=axes[0])
        sns.swarmplot(data=TP, color='black', ax=axes[0])
        sns.boxplot(data=TP, saturation=0.5, ax=axes[0])
        plt.setp(axes[0], alpha=0.3)
        axes[0].set_ylabel('True positive' + title_suffix)
        axes[0].set_title(title)
        sns.violinplot(data=FP, cut=True, linewidth=0.3, bw=0.5, ax=axes[1])
        sns.swarmplot(data=FP, color='black', ax=axes[1])
        sns.boxplot(data=FP, saturation=0.5, ax=axes[1])
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


def plotSnrResults(tr, title='', doPrint=True):
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
    plt.scatter(df.scienceSNR.values, df.ALstack_SNR.values, label='AL', color='b', alpha=0.4)
    plt.scatter(df.scienceSNR.values, df.ALstack_decorr_SNR.values, label='AL(decorr)', color='g', alpha=0.4)
    plt.scatter(df.scienceSNR.values, df.ZOGY_SNR.values, label='ZOGY', color='r', alpha=0.4)
    legend = plt.legend(loc='upper left', shadow=True)
    plt.xlabel('Science image SNR (measured)')
    plt.ylabel('Difference image SNR')
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
    sns.distplot(df.ALstack_decorr_SNR.values[~np.isnan(df.ALstack_decorr_SNR.values)], label='AL(decorr)', norm_hist=False)
    sns.distplot(df.ZOGY_SNR.values, label='ZOGY', norm_hist=False)
    sns.distplot(df.ZOGY_SNR.values, label='Science img (measured)', norm_hist=False)
    #sns.distplot(df.inputSNR.values, label='Input', norm_hist=False)
    plt.plot(np.repeat(df.inputSNR.values.mean(), 2), np.array([0, 0.4]), label='Input SNR', color='k')
    legend = plt.legend(loc='upper left', shadow=True)
    plt.xlabel('SNR')
    plt.ylabel('Frequency')
    plt.title(title)
    #df[['ZOGY_SNR', 'ALstack_SNR']].plot.hist(bins=20, alpha=0.5)

    plt.subplot(223)
    g = sns.violinplot(data=df[['ALstack_SNR', 'ALstack_decorr_SNR', 'ZOGY_SNR', 'scienceSNR', 'inputSNR']],
                       linewidth=0.3, bw=0.25, scale='width')
    #sns.swarmplot(data=df[['ALstack_SNR', 'ALstack_decorr_SNR', 'ZOGY_SNR', 'scienceSNR', 'inputSNR']],
    #              color='black', size=0.2, ax=g)
    plt.ylabel('SNR')
    g.set_xticklabels(g.get_xticklabels(), rotation=60)
    plt.title(title)

    #plt.figure(4)
    #plt.subplot(224)
    #plt.scatter([1], [1])

