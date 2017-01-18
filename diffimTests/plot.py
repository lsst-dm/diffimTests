import numpy as np


def zscale_image(input_img, contrast=0.25):
    """This emulates ds9's zscale feature. Returns the suggested minimum and
    maximum values to display."""

    samples = input_img.flatten()
    samples = samples[~np.isnan(samples)]
    samples.sort()
    chop_size = int(0.10*len(samples))
    subset = samples[chop_size:-chop_size]

    i_midpoint = int(len(subset)/2)
    I_mid = subset[i_midpoint]

    fit = np.polyfit(np.arange(len(subset)) - i_midpoint, subset, 1)
    # fit = [ slope, intercept]

    z1 = I_mid + fit[0]/contrast * (1-i_midpoint)/1.0
    z2 = I_mid + fit[0]/contrast * (len(subset)-i_midpoint)/1.0
    return z1, z2


def plotImageGrid(images, nrows_ncols=None, extent=None, clim=None, interpolation='none',
                  cmap='gray', imScale=2., cbar=True, titles=None, titlecol=['r', 'y'],
                  same_zscale=True, **kwds):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    from mpl_toolkits.axes_grid1 import ImageGrid
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    def add_inner_title(ax, title, loc, size=None, **kwargs):
        if size is None:
            size = dict(size=plt.rcParams['legend.fontsize'], color=titlecol[0])
        at = AnchoredText(title, loc=loc, prop=size,
                          pad=0., borderpad=0.5,
                          frameon=False, **kwargs)
        ax.add_artist(at)
        at.txt._text.set_path_effects([withStroke(foreground=titlecol[1], linewidth=3)])
        return at

    if nrows_ncols is None:
        tmp = np.int(np.floor(np.sqrt(len(images))))
        nrows_ncols = (tmp, np.int(np.ceil(np.float(len(images))/tmp)))
    if nrows_ncols[0] <= 0:
        nrows_ncols[0] = 1
    if nrows_ncols[1] <= 0:
        nrows_ncols[1] = 1
    size = (nrows_ncols[1]*imScale, nrows_ncols[0]*imScale)
    fig = plt.figure(1, size)
    igrid = ImageGrid(fig, 111,  # similar to subplot(111)
                      nrows_ncols=nrows_ncols, direction='row',  # creates 2x2 grid of axes
                      axes_pad=0.1,  # pad between axes in inch.
                      label_mode="L",  # share_all=True,
                      cbar_location="right", cbar_mode="single", cbar_size='3%')
    extentWasNone = False
    clim_orig = clim

    imagesToPlot = []

    for i in range(len(images)):
        ii = images[i]
        if hasattr(ii, 'computeImage'):
            if hasattr(ii, 'getDimensions'):
                img = afwImage.ImageD(ii.getDimensions())
                ii.computeImage(img, doNormalize=False)
                ii = img
            else:
                ii = ii.computeImage()
        if hasattr(ii, 'getImage'):
            ii = ii.getImage()
        if hasattr(ii, 'getMaskedImage'):
            ii = ii.getMaskedImage().getImage()
        if hasattr(ii, 'getArray'):
            bbox = ii.getBBox()
            if extent is None:
                extentWasNone = True
                extent = (bbox.getBeginX(), bbox.getEndX(), bbox.getBeginY(), bbox.getEndY())
            ii = ii.getArray()
        if extent is not None and not extentWasNone:
            ii = ii[extent[0]:extent[1], extent[2]:extent[3]]

        imagesToPlot.append(ii)

    if clim_orig is None and same_zscale:
        tmp_im = [iii.flatten() for iii in imagesToPlot]
        tmp_im = np.concatenate(tmp_im)
        clim = zscale_image(tmp_im)
        del tmp_im

    for i in range(len(imagesToPlot)):
        ii = imagesToPlot[i]
        if clim_orig is None:
            clim = zscale_image(ii)
        if cbar and clim_orig is not None:
            ii = np.clip(ii, clim[0], clim[1])
        if np.isclose(clim[0], clim[1]):
            clim = (clim[0], clim[1] + clim[0] / 10.)  # in case there's nothing in the image
        if np.isclose(clim[0], clim[1]):
            clim = (clim[0] - 0.1, clim[1] + 0.1)  # in case there's nothing in the image
        im = igrid[i].imshow(ii, origin='lower', interpolation=interpolation, cmap=cmap,
                             extent=extent, clim=clim, **kwds)
        if cbar:
            igrid[i].cax.colorbar(im)
        if titles is not None:  # assume titles is an array or tuple of same length as images.
            t = add_inner_title(igrid[i], titles[i], loc=2)
            t.patch.set_ec("none")
            t.patch.set_alpha(0.5)
        if extentWasNone:
            extent = None
        extentWasNone = False
    return igrid


# Code taken from https://github.com/lsst-dm/dmtn-006/blob/master/python/diasource_mosaic.py
def mosaicDIASources(repo_dir, visitid, ccdnum=10, cutout_size=30,
                     template_catalog=None, xnear=None, ynear=None, sourceIds=None, gridSpec=[7, 4],
                     dipoleFlag='ip_diffim_ClassificationDipole_value'):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    import matplotlib.gridspec as gridspec
    import lsst.daf.persistence as dafPersist

    #
    # This matches up which exposures were differenced against which templates,
    # and is purely specific to this particular set of data.
    if template_catalog is None:
        template_catalog = {197790: [197802, 198372, 198376, 198380, 198384],
                            197662: [198668, 199009, 199021, 199033],
                            197408: [197400, 197404, 197412],
                            197384: [197388, 197392],
                            197371: [197367, 197375, 197379]}
    # Need to invert this to template_visit_catalog[exposure] = template
    template_visit_catalog = {}
    for templateid, visits in template_catalog.iteritems():
        for visit in visits:
            template_visit_catalog[visit] = templateid

    def make_cutout(img, x, y, cutout_size=20):
        return img[(x-cutout_size//2):(x+cutout_size//2), (y-cutout_size//2):(y+cutout_size//2)]

    def group_items(items, group_length):
        for n in xrange(0, len(items), group_length):
            yield items[n:(n+group_length)]

    b = dafPersist.Butler(repo_dir)

    template_visit = template_visit_catalog[visitid]
    templateExposure = b.get("calexp", visit=template_visit, ccdnum=ccdnum, immediate=True)
    template_img, _, _ = templateExposure.getMaskedImage().getArrays()
    template_wcs = templateExposure.getWcs()

    sourceExposure = b.get("calexp", visit=visitid, ccdnum=ccdnum, immediate=True)
    source_img, _, _ = sourceExposure.getMaskedImage().getArrays()

    subtractedExposure = b.get("deepDiff_differenceExp", visit=visitid, ccdnum=ccdnum, immediate=True)
    subtracted_img, _, _ = subtractedExposure.getMaskedImage().getArrays()
    subtracted_wcs = subtractedExposure.getWcs()

    diaSources = b.get("deepDiff_diaSrc", visit=visitid, ccdnum=ccdnum, immediate=True)

    masked_img = subtractedExposure.getMaskedImage()
    img_arr, mask_arr, var_arr = masked_img.getArrays()
    z1, z2 = zscale_image(img_arr)

    top_level_grid = gridspec.GridSpec(gridSpec[0], gridSpec[1])

    source_ind = 0
    for source_n, source in enumerate(diaSources):

        source_id = source.getId()
        if sourceIds is not None and not np.in1d(source_id, sourceIds)[0]:
            continue

        source_x = source.get("ip_diffim_NaiveDipoleCentroid_x")
        source_y = source.get("ip_diffim_NaiveDipoleCentroid_y")
        if xnear is not None and not np.any(np.abs(source_x - xnear) <= cutout_size):
            continue
        if ynear is not None and not np.any(np.abs(source_y - ynear) <= cutout_size):
            continue

        #is_dipole = source.get("ip_diffim_ClassificationDipole_value") == 1
        dipoleLabel = ''
        if source.get(dipoleFlag) == 1:
            dipoleLabel = 'Dipole'
        if source.get("ip_diffim_DipoleFit_flag_classificationAttempted") == 1:
            dipoleLabel += ' *'
        template_xycoord = template_wcs.skyToPixel(subtracted_wcs.pixelToSky(source_x, source_y))
        cutouts = [make_cutout(template_img, template_xycoord.getY(), template_xycoord.getX(),
                               cutout_size=cutout_size),
                   make_cutout(source_img, source_y, source_x, cutout_size=cutout_size),
                   make_cutout(subtracted_img, source_y, source_x, cutout_size=cutout_size)]

        try:
            subgrid = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=top_level_grid[source_ind],
                                                       wspace=0)
        except:
            continue
        for cutout_n, cutout in enumerate(cutouts):
            plt.subplot(subgrid[0, cutout_n])
            plt.imshow(cutout, vmin=z1, vmax=z2, cmap=plt.cm.gray)
            plt.gca().xaxis.set_ticklabels([])
            plt.gca().yaxis.set_ticklabels([])

        plt.subplot(subgrid[0, 0])
        source_ind += 1
        #if is_dipole:
        #print(source_n, source_id)
        plt.ylabel(str(source_n) + dipoleLabel)


class PdfPages(object):
    def __init__(self, filename, **kwargs):
        from matplotlib.backends.backend_pdf import PdfPages as pdfp
        self.filename = filename
        self.pdf = pdfp(filename)

    def next(self):
        import matplotlib.pyplot as plt
        self.pdf.savefig()
        plt.close()

    def close(self):
        self.pdf.close()
