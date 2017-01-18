import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
#%matplotlib notebook
#import matplotlib.pylab as plt
import matplotlib
matplotlib.style.use('ggplot')
import warnings

import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
import seaborn as sns

import diffimTests as dit
reload(dit)

sns.set(style="whitegrid", palette="pastel", color_codes=True)

pd.options.display.max_columns = 9999
pd.set_option('display.width', 9999)

warnings.filterwarnings('ignore')


testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000),
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15.)
res = testObj.runTest(returnSources=True, matchDist=np.sqrt(1.5))
src = res['sources']
del res['sources']
print res


#from matplotlib.backends.backend_pdf import PdfPages
pdf = dit.PdfPages('tests.pdf')

cats = testObj.doForcedPhot(transientsOnly=True)
sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats

plt.scatter(sources['inputFlux_science']+10, fp_ZOGY['base_PsfFlux_flux']/fp_ZOGY['base_PsfFlux_fluxSigma'], label='ZOGY')
plt.scatter(sources['inputFlux_science'], fp_AL['base_PsfFlux_flux']/fp_AL['base_PsfFlux_fluxSigma'], label='AL', color='r')
plt.scatter(sources['inputFlux_science'], fp_ALd['base_PsfFlux_flux']/fp_ALd['base_PsfFlux_fluxSigma'], label='ALd', color='g')
plt.legend(loc='upper left')
plt.xlabel('input flux')
plt.ylabel('measured SNR')
plt.xlim(0, 2000)
pdf.next() #savefig(); plt.close()

cats = testObj.doForcedPhot(transientsOnly=False)
sources, fp1, fp2, fp_ZOGY, fp_AL, fp_ALd = cats

plt.scatter(sources['inputFlux_science']+10, fp_ZOGY['base_PsfFlux_flux']/fp_ZOGY['base_PsfFlux_fluxSigma'], label='ZOGY')
plt.scatter(sources['inputFlux_science'], fp_AL['base_PsfFlux_flux']/fp_AL['base_PsfFlux_fluxSigma'], label='AL', color='r', alpha=0.4)
plt.scatter(sources['inputFlux_science'], fp_ALd['base_PsfFlux_flux']/fp_ALd['base_PsfFlux_fluxSigma'], label='ALd', color='g')
plt.legend(loc='upper left')
plt.xlabel('input flux')
plt.ylabel('measured SNR')
plt.xlim(0, 20000)
pdf.next() #pdf.savefig(); plt.close()

meas = fp2['base_PsfFlux_flux']/fp2['base_PsfFlux_fluxSigma']
calc = testObj.im2.calcSNR(sources['inputFlux_science'], skyLimited=True)
print np.median(meas/calc)
plt.scatter(sources['inputFlux_science'], meas/calc)
plt.xlim(0, 20000)
plt.ylim(0.8, 1.2)
plt.xlabel('Input flux')
plt.ylabel('Measured SNR / Input SNR (science)')
pdf.next() #pdf.savefig(); plt.close()

testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000),
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=5.)


testObj.doPlotWithDetectionsHighlighted(transientsOnly=False, addPresub=True)
plt.xlim(0, 20010)
plt.ylim(-2, 205)
pdf.next() #pdf.savefig(); plt.close()

res, df = testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True)
plt.xlim(0, 2010)
plt.ylim(-0.2, 20)
pdf.next() #pdf.savefig(); plt.close()

tmp = df.ix[(df.scienceSNR > 12) & (df.scienceSNR < 15) & (df.inputFlux < 1500)]
testObj.doPlot(centroidCoord=[tmp.inputCentroid_y.values[0], tmp.inputCentroid_x.values[0]]);
pdf.next() #pdf.savefig(); plt.close()

testObj = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000),
                         varFlux2=np.linspace(200, 2000, 50),
                         #varFlux2=np.repeat(500., 50),
                         templateNoNoise=True, skyLimited=True,
                         avoidAllOverlaps=15.)

testObj.doPlotWithDetectionsHighlighted(transientsOnly=False, addPresub=True)
plt.xlim(0, 20010)
plt.ylim(-2, 205)
pdf.next() #pdf.savefig(); plt.close()

df = testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True)
plt.xlim(0, 2010)
plt.ylim(-0.2, 20)
pdf.next() #pdf.savefig(); plt.close()

testObj.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True,
                                        xaxisIsScienceForcedPhot=True)
plt.xlim(0, 2010)
plt.ylim(-2, 20)
pdf.next() #pdf.savefig(); plt.close()

testObj2 = dit.DiffimTest(n_sources=91, sourceFluxRange=(2000, 20000), 
                          varFlux2=np.repeat(610., 50),
                          templateNoNoise=True, skyLimited=True,
                          avoidAllOverlaps=15.)
testObj2.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True,
                                         xaxisIsScienceForcedPhot=True)
plt.xlim(400, 900)
plt.ylim(-0.2, 8);
pdf.next() #pdf.savefig(); plt.close()


reload(dit)
testObj3 = dit.DiffimTest(n_sources=500, sourceFluxRange=(2000, 20000), 
                         varFlux2=np.linspace(200, 2000, 50),
                         templateNoNoise=False, skyLimited=False,
                         avoidAllOverlaps=15.)
testObj3.doPlotWithDetectionsHighlighted(transientsOnly=True, addPresub=True,
                                         xaxisIsScienceForcedPhot=False)
plt.xlim(0, 2000)
plt.ylim(-2, 18);
pdf.next() #pdf.savefig(); plt.close()




pdf.close()
