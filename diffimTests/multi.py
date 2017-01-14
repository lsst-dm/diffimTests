import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed

from .diffimTests import DiffimTest


def getNumCores():
    num_cores = multiprocessing.cpu_count()
    if num_cores == 32:
        num_cores = 16
    elif num_cores == 8:
        num_cores = 3
    elif num_cores == 4:
        num_cores = 2
    print 'CORES:', num_cores


def runTest(flux, seed=66, n_varSources=50, n_sources=1000, **kwargs):
    sky = kwargs.get('sky', 300.)
    psf1 = kwargs.get('psf1', [1.6, 1.6])
    psf2 = kwargs.get('psf2', [1.8, 2.2])
    templateNoNoise = kwargs.get('templateNoNoise', True)
    skyLimited = kwargs.get('skyLimited', True)

    testObj = DiffimTest(sky=sky, psf1=psf1, psf2=psf2, varFlux2=np.repeat(flux, n_varSources),
                         n_sources=n_sources, sourceFluxRange=(200, 20000),
                         templateNoNoise=templateNoNoise, skyLimited=skyLimited, avoidAllOverlaps=15.,
                         seed=seed)
    res = testObj.runTest(returnSources=True)
    src = res['sources']
    del res['sources']
    res['flux'] = flux
    return res


# Using flux=600 for SNR=5 (see cell #4 of notebook '30. 4a. other psf models-real PSFs')
def runMultiDiffimTests(varSourceFlux=600., n_runs=100, **kwargs):
    inputs = [(f, seed) for f in [varSourceFlux] for seed in np.arange(66, 66+n_runs, 1)]
    print 'RUNNING:', len(inputs)
    num_cores = getNumCores()
    testResults = Parallel(n_jobs=num_cores, verbose=2)(delayed(runTest)(flux=i[0], seed=i[1], **kwargs)
                                                        for i in inputs)
    return testResults


methods = ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_decorr']


def plotResults(tr, doRates=False, title='', asHist=False, doPrint=True):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')

    import seaborn as sns
    sns.set(style="whitegrid", palette="pastel", color_codes=True)

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
