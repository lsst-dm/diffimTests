# This file can be imported into a jupyter notebook namespace by
#
# %run -n -i "plot_calexp_template_diffim.py"
#
# to use the individual functions. Also, there is a main function defined to be
# used as a stand alone script to provide easy reproducibility for certain
# figures.
#
# These functions are based on jupyter notebooks by Meredith.

import os
import numpy as np
import pandas as pd
import sqlite3
import lsst.daf.persistence as dafPersist
import lsst.geom
from astropy.visualization import (ZScaleInterval, AsinhStretch, ImageNormalize)
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.table import Table
import astropy.table
from matplotlib.backends.backend_pdf import PdfPages
import argparse


def loadAllPpdbObjects(repo, dbName='association.db'):
    """Load select DIAObject columns from a PPDB into a pandas dataframe.

    Parameters
    ----------
    repo : `str`
        Path to an output repository from an ap_pipe run.
    dbName : `str`, optional
        Name of the PPDB, which must reside in (or relative to) repo.

    Returns
    -------
    objTable : `pandas.DataFrame`
        DIA Object Table containing only objects with validityEnd NULL.
        Columns selected are presently hard-wired here.
    """
    connection = sqlite3.connect(os.path.join(repo, dbName))

    # These are the tables available in the ppdb
    tables = {'obj': 'DiaObject', 'src': 'DiaSource', 'ccd': 'CcdVisit'}

    # Only get objects with validityEnd NULL because that means
    # they are still valid
    objTable = pd.read_sql_query('select diaObjectId, ra, decl, nDiaSources, '
                                 'gPSFluxMean, validityEnd, flags from {0} where validityEnd is NULL;'.format(
                                     tables['obj']), connection)

    return objTable


def defMiniRegion(objTable):
    """Selects objects with RA, DEC coordinates within the "mini region"
    used by Meredith. This is an arbitrary fixed region for the Hits2015 data
    used for testing.
    """
    miniRegion = ((objTable['decl'] < -5.6) & (objTable['decl'] > -5.8) &
                  (objTable['ra'] > 155.2) & (objTable['ra'] < 155.3) &
                  (objTable['nDiaSources'] > 2))
    return miniRegion


def plotMiniRegion(objTable, miniRegion, title=None):
    print('Plotting {0} DIAObjects'.format(len(objTable.loc[miniRegion, 'ra'])))
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)
    ax1.scatter((objTable.loc[miniRegion, 'ra'].values*u.deg).to_value(u.rad),
                (objTable.loc[miniRegion, 'decl'].values*u.deg).to_value(u.rad),
                marker='.', lw=0, s=objTable.loc[miniRegion, 'nDiaSources']*8,
                c=objTable.loc[miniRegion, 'flags'], alpha=0.5)
    plt.xlabel('RA (rad)')
    plt.ylabel('Dec (rad)')
    plt.xlim([2.71040, 2.70875])
    plt.ylim([-0.1014, -0.0978])
    if title:
        plt.title(title)
# ---


def load_sources(repo, obj, sqliteFile='association.db'):
    connection = sqlite3.connect(os.path.join(repo, sqliteFile))
    tables = {'obj': 'DiaObject', 'src': 'DiaSource', 'ccd': 'CcdVisit'}
    srcTable = pd.read_sql_query('select diaSourceId, diaObjectId, ccdVisitId, midPointTai, apFlux,'
                                 ' psFlux, apFluxErr, psFluxErr, totFlux, totFluxErr, flags '
                                 'from {} where diaObjectId = ?'.format(tables['src']), connection,
                                 params=(obj, ))
    connection.close()
    return(srcTable)


def get_dataIdDict(ccdVisitId):
    return {'visit': ccdVisitId // 100, 'ccdnum': ccdVisitId % 100}


def plot_images(repo, templateRepo, obj, patch, objTable, cutoutIdx=0,
                plotAllCutouts=False, diffimType='deepDiff_differenceExp', pdfWriter=None):
    sources = Table.from_pandas(load_sources(repo, obj))
    sources.sort(keys='ccdVisitId')

    ra = objTable.loc[objTable['diaObjectId'] == obj, 'ra']
    dec = objTable.loc[objTable['diaObjectId'] == obj, 'decl']
    flags = sources['flags']
#    dataIds = sources['ccdVisitId'].values  # these are ints
#    srcIds = list(sources['diaSourceId'].values)
#    dataIdDicts = []
#    for dataId in dataIds:
#        visit = dataId // 100
#        ccdnum = dataId % 100
#        dataIdDict = {'visit': visit, 'ccdnum': ccdnum}
#        dataIdDicts.append(dataIdDict)
    centerSource = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    size = lsst.geom.Extent2I(100, 100)

    print('DIAObject ID:', obj)
    print('Flags:', flags)
    print('RA (deg):', ra.values)
    print('Dec (deg):', dec.values)
    print('DIASource IDs:', sources['diaSourceId'])
    print('Data IDs:', sources['ccdVisitId'])

    fig = plt.figure(figsize=(10, 4))
    fig.suptitle('DIAObject ID:{}'.format(obj))

    # First image is for summary

    # processed image
    ax = fig.add_subplot(1, 3, 1)
    fig.subplots_adjust(left=0.05, right=0.93, bottom=0.05, wspace=0.15)

    dataIdDict = get_dataIdDict(sources['ccdVisitId'][cutoutIdx])
    butler = dafPersist.Butler(repo)
    calexpFirst = butler.get('calexp', dataIdDict)
    ax.set_title('{visit} {ccdnum:02d}'.format(**dataIdDict))
    calexpCutout = calexpFirst.getCutout(centerSource, size)
    bbox = calexpCutout.getBBox()
    extentR = (bbox.getMaxY()+0.5, bbox.getMinY()-0.5, bbox.getMaxX()+0.5, bbox.getMinX()-0.5)
    calexpArray = calexpCutout.getMaskedImage().getImage().getArray()
    calexpNorm = ImageNormalize(calexpArray, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(calexpArray.T[::-1, ::-1], origin='lower', cmap='gray', norm=calexpNorm, extent=extentR)
    ax.grid(True)

    # template image
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title(patch)
    templateDataId = {'filter': 'g', 'tract': 0, 'patch': patch}
    butlerTemplate = dafPersist.Butler(templateRepo)
    template = butlerTemplate.get('deepCoadd', dataId=templateDataId)
    templateCutout = template.getCutout(centerSource, size)
    bbox = templateCutout.getBBox()
    # The template seems to have the usual orientation by default
    extent = (bbox.getMinX()-0.5, bbox.getMaxX()+0.5, bbox.getMinY()-0.5, bbox.getMaxY()+0.5)
    templateArray = templateCutout.getMaskedImage().getImage().getArray()
    templateNorm = ImageNormalize(templateArray, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(templateArray, origin='lower', cmap='Blues_r', norm=templateNorm, extent=extent)
    ax.grid(True)

    # difference image
    ax = fig.add_subplot(1, 3, 3)

    vnums = np.arange(len(sources), dtype=int)

    # 'totflux' can be a MaskedColumns if there are masked values only
    if isinstance(sources['totFlux'], astropy.table.MaskedColumn):
        ax.errorbar(
            vnums, sources['totFlux'].filled(
                np.nan), yerr=sources['totFluxErr'].filled(
                np.nan), fmt='o', color='black')
    else:
        ax.errorbar(vnums, sources['totFlux'], yerr=sources['totFluxErr'], fmt='o', color='black')
    ax2 = ax.twinx()
    # Plotting gives error if all values are masked and fill_value
    if isinstance(sources['psFlux'], astropy.table.MaskedColumn):
        ax2.errorbar(
            vnums, sources['psFlux'].filled(
                np.nan), yerr=sources['psFluxErr'].filled(
                np.nan), fmt='x', color='blue')
    else:
        ax2.errorbar(vnums, sources['psFlux'], yerr=sources['psFluxErr'], fmt='x', color='blue')
#    ax2.plot (vnums, sources['psFlux'], '.', color='blue')

    if pdfWriter is not None:
        pdfWriter.savefig(fig)
        plt.close(fig)

    n_plots_per_fig = 4
    n_figs = len(sources) // 4
    if len(sources) % 4 != 0:
        n_figs += 1
    fig_idx = 1

    if plotAllCutouts:
        fig = None

        n_panels = len(sources)
        panel_idx = 1
        i_source = 0
        while i_source < len(sources):
            row = sources[i_source]
            if fig is None:
                fig = plt.figure(figsize=(10, 5.5))
                fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, wspace=0.15, hspace=0.1)
                fig.suptitle('DIAObject {}; {}/{}'.format(obj, fig_idx, n_figs))
                fig_idx += 1
                splot_idx = 1

            dataIdDict = get_dataIdDict(row['ccdVisitId'])
            srcId = row['diaSourceId']

            # Upper row calexp
            ax = fig.add_subplot(2, n_plots_per_fig, splot_idx)
            calexpFirst = butler.get('calexp', dataIdDict)
            ax.set_title('{visit} {ccdnum:02d}; {idx}/{n_panels}'.format(idx=panel_idx,
                                                                         n_panels=n_panels, **dataIdDict))
            calexpCutout = calexpFirst.getCutout(centerSource, size)
            bbox = calexpCutout.getBBox()
            extentR = (bbox.getMaxY()+0.5, bbox.getMinY()-0.5, bbox.getMaxX()+0.5, bbox.getMinX()-0.5)
            calexpArray = calexpCutout.getMaskedImage().getImage().getArray()
            calexpNorm = ImageNormalize(calexpArray, interval=ZScaleInterval(), stretch=AsinhStretch())
            ax.imshow(calexpArray.T[::-1, ::-1], origin='lower',
                      cmap='gray', norm=calexpNorm, extent=extentR)
            ax.get_xaxis().set_visible(False)
            ax.text(
                0.1,
                0.9,
                '{:.0f}'.format(
                    row['totFlux']),
                transform=ax.transAxes,
                backgroundcolor='lightgrey')

            # Bottom row diffim
            ax = fig.add_subplot(2, n_plots_per_fig, splot_idx + n_plots_per_fig)
            ax.set_title('{}'.format(srcId))
            diffimFirst = butler.get(diffimType, dataIdDict)
            diffimCutout = diffimFirst.getCutout(centerSource, size)
            bbox = diffimCutout.getBBox()
            extentR = (bbox.getMaxY()+0.5, bbox.getMinY()-0.5, bbox.getMaxX()+0.5, bbox.getMinX()-0.5)
            diffimArray = diffimCutout.getMaskedImage().getImage().getArray()
            diffimNorm = ImageNormalize(diffimArray, interval=ZScaleInterval(), stretch=AsinhStretch())

            ax.imshow(diffimArray.T[::-1, ::-1], origin='lower',
                      cmap='gray', norm=diffimNorm, extent=extentR)
            ax.text(
                0.1,
                0.9,
                '{:.0f}'.format(
                    row['psFlux']),
                transform=ax.transAxes,
                backgroundcolor='lightsteelblue')

            panel_idx += 1
            splot_idx += 1
            if splot_idx > n_plots_per_fig:
                if pdfWriter is not None:
                    pdfWriter.savefig(fig)
                    plt.close(fig)
                fig = None

            i_source += 1

        if fig is not None and pdfWriter is not None:
            pdfWriter.savefig(fig)
            plt.close(fig)


def patchFinder(obj, objTable, templateButler, patchList):
    """Brute force patch finder for obj RA,DEC"""
    for patch in patchList:
        ra = objTable.loc[objTable['diaObjectId'] == obj, 'ra']
        dec = objTable.loc[objTable['diaObjectId'] == obj, 'decl']
        centerSource = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
        size = lsst.geom.Extent2I(30, 30)
        templateDataId = {'filter': 'g', 'tract': 0, 'patch': patch}
        templateImage = templateButler.get('deepCoadd', dataId=templateDataId)
        try:
            templateImage.getCutout(centerSource, size)
        except Exception:
            continue
        else:
            templatePatch = patch
#             print('template patch:', templatePatch)
#             print('object id:', obj)
            return templatePatch
            break

# ==============
# Argument processing and standalone main function


def parse_args():
    """Command line arguments. Specify a case identifier or repo and output
    details.
    """

    parser = argparse.ArgumentParser(description="Plots calexps and templates along with difference images "
                                     "for objects read from the ppdb of the given repository. Specify either "
                                     "a --case or the repositories.")
    parser.add_argument("--case", help="Hard-wired case identifier for reproducibility. "
                        "See docstrings and code notes for more details ", choices=["proc_2019-02-21", ])
    parser.add_argument("--calexp_repo", help="Butler repo or rerun directory of science exposures "
                        "(calexps) to use.")
    parser.add_argument("--template_repo", help="Butler repo for templates (deepCoadd) to use.")
    parser.add_argument("--pdfName", help="If specified, matplotlib figures are written into pdf files "
                        "with this base name")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.case == "proc_2019-02-21":
        cwpRepo = '/home/gkovacs/data/repo_DM-17825/ingested/rerun/proc_2019-02-21'
        cwpTemplateRepo = '/home/gkovacs/data/repo_DM-17825/templates'
        pdfName = 'proc_2019-02-21_diffims_mini'
    elif args.case is None:
        cwpRepo = args.calexp_repo
        cwpTemplateRepo = args.template_repo
        pdfName = args.pdfName
        if not cwpRepo or not cwpTemplateRepo:
            raise RuntimeError("If no --case given, --calexp_repo and --template_repo must be specified.")

    import matplotlib
    matplotlib.use('Qt5Agg')

    butlerCwpTemplate = dafPersist.Butler(cwpTemplateRepo)

    patchList = ['10,8', '11,8', '12,8', '13,8',
                 '10,7', '11,7', '12,7', '13,7',
                 '10,9', '11,9', '12,9', '13,9',
                 '10,5', '11,5', '12,5', '13,5',
                 '10,6', '11,6', '12,6', '13,6',
                 '10,10', '11,10', '12,10', '13,10']

    cwpObjTable = loadAllPpdbObjects(cwpRepo)
    cwpMiniRegion = defMiniRegion(cwpObjTable)
    cwpMiniUnflagged = cwpMiniRegion & (cwpObjTable['flags'] == 0)
    cwpObjList = list(cwpObjTable.loc[cwpMiniRegion, 'diaObjectId'])
    cwpObjList.sort()

    # Find the patch that belongs to the mini region
    patch = patchFinder(cwpObjTable.loc[cwpMiniUnflagged, 'diaObjectId'].values[0],
                        cwpObjTable, butlerCwpTemplate, patchList)

    if pdfName:
        W = PdfPages('{}_all.pdf'.format(pdfName))
    else:
        W = None

    try:
        for obj in cwpObjList:
            print(obj)
            plot_images(cwpRepo, cwpTemplateRepo, obj, patch, cwpObjTable, plotAllCutouts=True, pdfWriter=W)
    finally:
        if W is not None:
            W.close()

    cwpObjList = list(cwpObjTable.loc[cwpMiniUnflagged, 'diaObjectId'])
    cwpObjList.sort()

    if pdfName:
        W = PdfPages('{}_unflagged.pdf'.format(pdfName))
    else:
        W = None

    try:
        for obj in cwpObjList:
            print(obj)
            plot_images(cwpRepo, cwpTemplateRepo, obj, patch, cwpObjTable, plotAllCutouts=True, pdfWriter=W)
    finally:
        if W is not None:
            W.close()


if __name__ == "__main__":
    main()
