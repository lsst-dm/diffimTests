import numpy as np
import scipy.stats
import pandas as pd

try:
    import lsst.afw.table as afwTable
    import lsst.afw.geom as afwGeom
    import lsst.afw.detection as afwDetection
except:
    pass

# compute rms x- and y- pixel offset between two catalogs. Assume input is 2- or 3-column dataframe.
# Assume 1st column is x-coord and 2nd is y-coord. 
# If 3-column then 3rd column is flux and use flux**2 as weighting on shift calculation
# We need some severe filtering if we have lots of sources

# TBD: use afwTable.matchXy(src1, src2, matchDist)
# https://github.com/lsst/meas_astrom/blob/master/include/lsst/meas/astrom/makeMatchStatistics.h
def computeOffsets(src1, src2, threshold=2.5, fluxWeighted=True, returnOffsetsOnly=False):
    dist = np.sqrt(np.add.outer(src1.iloc[:, 0], -src2.iloc[:, 0])**2. +
                   np.add.outer(src1.iloc[:, 1], -src2.iloc[:, 1])**2.)  # in pixels
    matches = np.where(dist <= threshold)
    match1 = src1.iloc[matches[0], :]
    match2 = src2.iloc[matches[1], :]
    if len(matches[0]) > src1.shape[0]:
        print 'WARNING: Threshold for ast. matching is probably too small:', match1.shape[0], src1.shape[0]
    if len(matches[1]) > src2.shape[0]:
        print 'WARNING: Threshold for ast. matching is probably too small:', match2.shape[0], src2.shape[0]
    dx = (match1.iloc[:, 0].values - match2.iloc[:, 0].values)
    _, dxlow, dxupp = scipy.stats.sigmaclip(dx, low=2, high=2)
    dy = (match1.iloc[:, 1].values - match2.iloc[:, 1].values)
    _, dylow, dyupp = scipy.stats.sigmaclip(dy, low=2, high=2)
    inds = (dx >= dxlow) & (dx <= dxupp) & (dy >= dylow) & (dy <= dyupp)
    if np.sum(inds) <= 0:
        inds = np.repeat(True, len(inds))
    weights = np.ones(inds.sum())
    if fluxWeighted and match1.shape[1] >= 3:
        fluxes = (match1.iloc[:, 2].values + match2.iloc[:, 2].values) / 2.
        weights = fluxes[inds]**2.
    rms = dx[inds]**2. + dy[inds]**2.
    if returnOffsetsOnly:
        return dx[inds], dy[inds]
    if np.sum(weights) == 0.:
        weights = np.zeros(len(rms))
    dx = np.average(np.abs(dx[inds]**2.), weights=weights)
    dy = np.average(np.abs(dy[inds]**2.), weights=weights)
    rms = np.average(rms, weights=weights)
    return dx, dy, rms


### Catalog utilities below!

def catalogToDF(cat):
    return pd.DataFrame({col: cat.columns[col] for col in cat.schema.getNames()})

# This is NOT functional for all catalogs (i.e., catalog -> DF works, but catalog -> DF -> catalog may not.
def dfToCatalog(df, centroidSlot='centroid'):
    schema = afwTable.SourceTable.makeMinimalSchema()
    if centroidSlot is not None:
        centroidKey = afwTable.Point2DKey.addFields(schema, centroidSlot, centroidSlot, 'pixel')
        schema.getAliasMap().set('slot_Centroid', centroidSlot)
    for col in df.columns.values:
        dt = df[col].dtype.type
        if df[col].dtype.name == 'bool':  # booleans and int64 not supported in tables?
            dt = int
        elif df[col].dtype.name == 'int64' or df[col].dtype.name == 'long':
            dt = long
        try:
            schema.addField(col, type=dt, doc=col)
        except Exception as e:
            pass
    table = afwTable.SourceTable.make(schema)
    sources = afwTable.SourceCatalog(table)

    for index, row in df.iterrows():
        record = sources.addNew()
        for col in df.columns.values:
            val = row[col]
            record.set(col, val)

    sources = sources.copy(deep=True)  # make it contiguous
    return sources

# Centroids is a 4-column matrix with x, y, flux(template), flux(science)
# transientsOnly means that sources with flux(template)==0 are skipped.
def centroidsToCatalog(centroids, expWcs, transientsOnly=False):
    schema = afwTable.SourceTable.makeMinimalSchema()
    centroidKey = afwTable.Point2DKey.addFields(schema, 'centroid', 'centroid', 'pixel')
    schema.getAliasMap().set('slot_Centroid', 'centroid')
    #schema.addField('centroid_x', type=float, doc='x pixel coord')
    #schema.addField('centroid_y', type=float, doc='y pixel coord')
    schema.addField('inputFlux_template', type=float, doc='input flux in template')
    schema.addField('inputFlux_science', type=float, doc='input flux in science image')
    table = afwTable.SourceTable.make(schema)
    sources = afwTable.SourceCatalog(table)

    footprint_radius = 5  # pixels

    for row in centroids:
        if transientsOnly and row[2] != 0.:
            continue
        record = sources.addNew()
        coord = expWcs.pixelToSky(row[0], row[1])
        record.setCoord(coord)
        record.set(centroidKey, afwGeom.Point2D(row[0], row[1]))
        record.set('inputFlux_template', row[2])
        record.set('inputFlux_science', row[3])

        fpCenter = afwGeom.Point2I(afwGeom.Point2D(row[0], row[1])) #expWcs.skyToPixel(coord))
        footprint = afwDetection.Footprint(fpCenter, footprint_radius)
        record.setFootprint(footprint)

    sources = sources.copy(deep=True)  # make it contiguous
    return sources
