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


def runTest(flux, seed=66, sky=300., n_sources=1000, n_varSources=50):
    global_psf1 = [1.6, 1.6]
    global_psf2 = [1.8, 2.2]

    testObj = DiffimTest(sky=sky, psf1=global_psf1, psf2=global_psf2,
                         varFlux2=np.repeat(flux, n_varSources), variablesNearCenter=False,
                         n_sources=n_sources, sourceFluxRange=(200, 20000),
                         templateNoNoise=True, skyLimited=True, avoidAllOverlaps=15.,
                         seed=seed)
    res = testObj.runTest(returnSources=True)
    src = res['sources']
    del res['sources']
    res['flux'] = flux
    return res


# Using flux=600 for SNR=5 (see cell #4 of notebook '30. 4a. other psf models-real PSFs')
def runMultiDiffimTests(varSourceFlux=600., n_runs=100):
    inputs = [(f, seed) for f in [varSourceFlux] for seed in np.arange(66, 66+n_runs, 1)]
    print 'RUNNING:', len(inputs)
    num_cores = getNumCores()
    testResults = Parallel(n_jobs=num_cores, verbose=2)(delayed(runTest)(i[0], i[1]) for i in inputs)
    return testResults


def plotResults(tr):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')

    import seaborn as sns
    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    methods = ['ALstack', 'ZOGY', 'SZOGY', 'ALstack_noDecorr']
    FN = pd.DataFrame({key: np.array([t[key]['FN'] for t in tr]) for key in methods})
    FP = pd.DataFrame({key: np.array([t[key]['FP'] for t in tr]) for key in methods})
    TP = pd.DataFrame({key: np.array([t[key]['TP'] for t in tr]) for key in methods})
    print 'FN:', '\n', FN.mean()
    print 'FP:', '\n', FP.mean()
    print 'TP:', '\n', TP.mean()

    matplotlib.rcParams['figure.figsize'] = (18.0, 6.0)
    fig, axes = plt.subplots(nrows=1, ncols=2)

    sns.violinplot(data=TP, inner="quart", cut=True, linewidth=0.3, bw=0.5, ax=axes[0])
    axes[0].set_title('True positives')
    #axes[0].set_ylim((0, 31))
    sns.violinplot(data=FP, inner="quart", cut=True, linewidth=0.3, bw=0.5, ax=axes[1])
    axes[1].set_title('False positives')
    axes[1].set_ylim((-1, 40))
