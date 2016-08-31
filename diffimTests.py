import numpy as np
import scipy
import scipy.stats
from scipy.fftpack import fft2, ifft2, fftfreq, fftshift

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath

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
                  cmap='gray', imScale=2., cbar=True, titles=None, titlecol=['r','y']):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    from mpl_toolkits.axes_grid1 import ImageGrid

    def add_inner_title(ax, title, loc, size=None, **kwargs):
        from matplotlib.offsetbox import AnchoredText
        from matplotlib.patheffects import withStroke
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
                      cbar_location="right", cbar_mode="single", cbar_size='7%')
    extentWasNone = False
    clim_orig = clim
    for i in range(len(images)):
        ii = images[i]
        if hasattr(ii, 'computeImage'):
            img = afwImage.ImageD(ii.getDimensions())
            ii.computeImage(img, doNormalize=False)
            ii = img
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
        if clim_orig is None:
            clim = zscale_image(ii)
        if cbar and clim_orig is not None:
            ii = np.clip(ii, clim[0], clim[1])
        im = igrid[i].imshow(ii, origin='lower', interpolation=interpolation, cmap=cmap,
                             extent=extent, clim=clim)
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

def gaussian2d(grid, m=None, s=None):
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

    if m is None:
        m = [0., 0.]
    if s is None:
        s = [1., 1.]
    cov = [[s[0], 0], [0, s[1]]]
    var = scipy.stats.multivariate_normal(mean=m, cov=cov)
    return var.pdf(grid)

def singleGaussian2d(x, y, xc, yc, sigma_x=1., sigma_y=1., theta=0., offset=0.):
    theta = (theta/180.) * np.pi
    cos_theta2, sin_theta2 = np.cos(theta)**2., np.sin(theta)**2.
    sigma_x2, sigma_y2 = sigma_x**2., sigma_y**2.
    a = cos_theta2/(2.*sigma_x2) + sin_theta2/(2.*sigma_y2)
    b = -(np.sin(2.*theta))/(4.*sigma_x2) + (np.sin(2.*theta))/(4.*sigma_y2)
    c = sin_theta2/(2.*sigma_x2) + cos_theta2/(2.*sigma_y2)
    xxc, yyc = x-xc, y-yc
    out = np.exp( - (a*(xxc**2.) + 2.*b*xxc*yyc + c*(yyc**2.)))
    if offset != 0.:
        out += offset
    out /= out.sum()
    return out

# Make the two "images". im1 is the template, im2 is the science
# image.
# NOTE: having sources near the edges really messes up the
# fitting (probably because of the convolution). So make sure no
# sources are near the edge.
# NOTE: also it seems that having the variable source with a large
# flux increase also messes up the fitting (seems to lead to
# overfitting -- perhaps to the source itself). This might be fixed by
# adding more constant sources.

def makeFakeImages(xim=None, yim=None, sig1=0.2, sig2=0.2, psf1=None, psf2=None, offset=None,
                   psf_yvary_factor=0.2, varSourceChange=1/50., theta1=0., theta2=-45., im2background=10.,
                   n_sources=500, seed=66):
    np.random.seed(seed)

    # psf1 = 1.6 # sigma in pixels im1 will be template
    # psf2 = 2.2 # sigma in pixels im2 will be science image. make the psf in this image slighly offset and elongated
    psf1 = [1.6, 1.6] if psf1 is None else psf1
    print 'Template PSF:', psf1, theta1
    psf2 = [1.8, 2.2] if psf2 is None else psf2
    print 'Science PSF:', psf2, theta2
    print np.sqrt(psf2[0]**2 - psf1[1]**2)
    # offset = 0.2  # astrometric offset (pixels) between the two images
    offset = [0.2, 0.2] if offset is None else offset
    print 'Offset:', offset

    xim = np.arange(-256, 256, 1) if xim is None else xim
    yim = xim.copy() if yim is None else yim
    x0im, y0im = np.meshgrid(xim, yim)
    fluxes = np.random.uniform(50, 30000, n_sources)
    xposns = np.random.uniform(xim.min()+16, xim.max()-5, n_sources)
    yposns = np.random.uniform(yim.min()+16, yim.max()-5, n_sources)

    # Make the source closest to the center of the image the one that increases in flux
    ind = np.argmin(xposns**2. + yposns**2.)
    #print ind, xposns[ind], yposns[ind]

    im1 = np.random.normal(scale=sig1, size=x0im.shape)  # sigma of template
    im2 = np.random.normal(scale=sig2, size=x0im.shape)  # sigma of science image

    psf2_yvary = psf_yvary_factor * (yim.mean() - yposns) / yim.max()  # variation in y-width of psf in science image across (x-dim of) image
    print 'PSF y spatial-variation:', psf2_yvary.min(), psf2_yvary.max()
    # psf2_yvary[:] = 1.1  # turn it off for now, just add a constant 1.1 pixel horizontal width

    for i in range(n_sources):
        flux = fluxes[i]
        tmp1 = flux * singleGaussian2d(x0im, y0im, xposns[i], yposns[i], psf1[0], psf1[1], theta=theta1)
        im1 += tmp1
        if i == ind:
            flux += flux * varSourceChange  # / 50.
        tmp2 = flux * singleGaussian2d(x0im, y0im, xposns[i]+offset[0], yposns[i]+offset[1],
                                       psf2[0], psf2[1]+psf2_yvary[i], theta=theta2)
        im2 += tmp2

    # Add a (constant, for now) background offset to im2
    if im2background != 0.:  # im2background = 10.
        print 'Background:', im2background
        im2 += im2background

    im1_psf = singleGaussian2d(x0im, y0im, 0, 0, psf1[0], psf1[1], theta=theta1)
    im2_psf = singleGaussian2d(x0im, y0im, offset[0], offset[1], psf2[0], psf2[1], theta=theta2)
    return im1, im2, im1_psf, im2_psf

# Okay, here we start the A&L basis functions...
# Update: it looks like the actual code in the stack uses chebyshev1 polynomials!
# Note these are essentially the same but with different scale factors.

# Here beta is a rescale factor but this is NOT what it is really used for.
# Can be used to rescale so that sigGauss[1] = sqrt(sigmaPsf_I^2 - sigmaPsf_T^2)
def chebGauss2d(x, y, m=None, s=None, ord=[0,0], beta=1., verbose=False):
    from numpy.polynomial.chebyshev import chebval2d
    if m is None:
        m = [0., 0.]
    if s is None:
        s = [1., 1.]
    # cov = [[s[0], 0], [0, s[1]]]
    coefLen = np.max(ord)+1
    coef0 = np.zeros(coefLen)
    coef0[ord[0]] = 1
    coef1 = np.zeros(coefLen)
    coef1[ord[1]] = 1
    if verbose:
        print s, beta, ord, coef0, coef1
    ga = singleGaussian2d(x, y, 0, 0, s[0]/beta, s[1]/beta)
    ch = chebval2d(x, y, c=np.outer(coef0, coef1))
    return ch * ga

# use same parameters as from the stack.
# TBD: is a degGauss of 2 mean it goes up to order 2 (i.e. $x^2$)? Or
# is it 2 orders so it only goes up to linear ($x$)? Probably the
# former, so that's what we'll use.

# Parameters from stack
# betaGauss is actually the scale factor for sigGauss -> sigGauss[0] = sigGauss[1]/betaGauss and
#   sigGauss[2] = sigGauss[1] * betaGauss. Looks like here, betaGauss is 2 (see sigGauss below) but
#   we'll just set it to 1.
# Note should rescale sigGauss so sigGauss[1] = sqrt(sigma_I^2 - sigma_T^2)
# betaGauss = 1   # in the Becker et al. paper betaGauss is 1 but PSF is more like 2 pixels?
# sigGauss = [0.75, 1.5, 3.0]
# degGauss = [4, 2, 2]
# # Parameters from and Becker et al. (2012)
# degGauss = [6, 4, 2]

def getALChebGaussBases(x0, y0, sigGauss=None, degGauss=None, betaGauss=1, verbose=True):
    sigGauss = [0.75, 1.5, 3.0] if sigGauss is None else sigGauss
    #degGauss = [4, 2, 2] if degGauss is None else degGauss
    degGauss = [6, 4, 2] if degGauss is None else degGauss

    # Old, too many bases:
    #basis = [chebGauss2d(x0, y0, grid, m=[0,0], s=[sig0,sig1], ord=[deg0,deg1], beta=betaGauss, verbose=False) for i0,sig0 in enumerate(sigGauss) for i1,sig1 in enumerate(sigGauss) for deg0 in range(degGauss[i0]) for deg1 in range(degGauss[i1])]

    def get_valid_inds(Nmax):
        tmp = np.add.outer(range(Nmax+1), range(Nmax+1))
        return np.where(tmp <= Nmax)

    inds = [get_valid_inds(i) for i in degGauss]
    if verbose:
        for i in inds:
            print i
    basis = [chebGauss2d(x0, y0, m=[0,0], s=[sig,sig], ord=[inds[i][0][ind], inds[i][1][ind]], beta=betaGauss, verbose=verbose) for i,sig in enumerate(sigGauss) for ind in range(len(inds[i][0]))]
    return basis

# Convolve im1 (template) with the basis functions, and make these the *new* bases.
# Input 'basis' is the output of getALChebGaussBases().

def makeImageBases(im1, basis):
    import scipy.signal
    #basis2 = [scipy.signal.fftconvolve(im1, b, mode='same') for b in basis]
    basis2 = [scipy.ndimage.filters.convolve(im1, b, mode='constant') for b in basis]
    return basis2

def makeSpatialBases(im1, basis, basis2, spatialKernelOrder=2, spatialBackgroundOrder=2, verbose=False):
    # Then make the spatially modified basis by simply multiplying the constant
    # basis (basis2 from makeImageBases()) by a polynomial along the image coordinate.
    # Note that since we *are* including i=0, this new basis *does include* basis2 and
    # thus can replace it.

    # Here beta is a rescale factor but this is NOT what it is really used for.
    # Can be used to rescale so that sigGauss[1] = sqrt(sigmaPsf_I^2 - sigmaPsf_T^2)

    # Apparently the stack uses a spatial kernel order of 2? (2nd-order?)
    #spatialKernelOrder = 2  # 0
    # Same for background order.
    #spatialBackgroundOrder = 2

    def cheb2d(x, y, ord=[0,0], verbose=False):
        from numpy.polynomial.chebyshev import chebval2d
        coefLen = np.max(ord)+1
        coef0 = np.zeros(coefLen)
        coef0[ord[0]] = 1
        coef1 = np.zeros(coefLen)
        coef1[ord[1]] = 1
        if verbose:
            print ord, coef0, coef1
        ch = chebval2d(x, y, c=np.outer(coef0, coef1))
        return ch

    def get_valid_inds(Nmax):
        tmp = np.add.outer(range(Nmax+1), range(Nmax+1))
        return np.where(tmp <= Nmax)

    spatialBasis = bgBasis = None

    spatialInds = get_valid_inds(spatialKernelOrder)
    if verbose:
        print spatialInds

    #xim = np.arange(np.int(-np.floor(im1.shape[0]/2.)), np.int(np.floor(im1.shape[0]/2)))
    #yim = np.arange(np.int(-np.floor(im1.shape[1]/2.)), np.int(np.floor(im1.shape[1]/2)))
    x0im, y0im = getImageGrid(im1) #np.meshgrid(xim, yim)

    # Note the ordering of the loop is important! Make the basis2 the last one so the first set of values
    # that are returned are all of the original (basis2) unmodified bases.
    # Store "spatialBasis" which is the kernel basis and the spatial basis separated so we can recompute the
    # final kernel at the end. Include in index 2 the "original" kernel basis as well.
    if spatialKernelOrder > 0:
        spatialBasis = [[basis2[bi], cheb2d(x0im, y0im, ord=[spatialInds[0][i], spatialInds[1][i]], verbose=False), basis[bi]] for i in range(1,len(spatialInds[0])) for bi in range(len(basis2))]
        #basis2m = [b * cheb2d(x0im, y0im, ord=[spatialInds[0][i], spatialInds[1][i]], verbose=False) for i in range(1,len(spatialInds[0])) for b in basis2]

    spatialBgInds = get_valid_inds(spatialBackgroundOrder)
    if verbose:
        print spatialBgInds

    # Then make the spatial background part
    if spatialBackgroundOrder > 0:
        bgBasis = [cheb2d(x0im, y0im, ord=[spatialBgInds[0][i], spatialBgInds[1][i]], verbose=False) for i in range(len(spatialBgInds[0]))]

    return spatialBasis, bgBasis

# Collect the bases into a single matrix
# ITMT, let's make sure all the bases are on a reasonable scale.
def collectAllBases(basis2, spatialBasis, bgBasis, verbose=False):

    basis2a = np.vstack([b.flatten() for b in basis2]).T

    constKernelIndices = np.arange(0, basis2a.shape[1])
    if verbose:
        print constKernelIndices

    nonConstKernelIndices = None
    if spatialBasis is not None:
        b1 = np.vstack([(b[0]*b[1]).flatten() for b in spatialBasis]).T
        nonConstKernelIndices = np.arange(basis2a.shape[1], basis2a.shape[1]+b1.shape[1])
        basis2a = np.hstack([basis2a, b1])
        if verbose:
            print nonConstKernelIndices

    bgIndices = None
    if bgBasis is not None:
        b1 = np.vstack([b.flatten() for b in bgBasis]).T
        bgIndices = np.arange(basis2a.shape[1], basis2a.shape[1]+b1.shape[1])
        basis2a = np.hstack([basis2a, b1])
        if verbose:
            print bgIndices

    # Rescale the bases so that the "standard" A&L linear fit will work (i.e. when squared, not too large!)
    basisOffset = 0.  # basis2a.mean(0) + 0.1
    basisScale = basis2a.std(0) + 0.1    # avoid division by zero
    basis2a = (basis2a-basisOffset)/(basisScale)
    return basis2a, (constKernelIndices, nonConstKernelIndices, bgIndices), (basisOffset, basisScale)

# Do the linear fit to compute the matching kernel. This is NOT the
# same fit as is done by standard A&L but gives the same results. This
# will not work for very large images. See below. The resulting fit is
# the matched template.

def doTheLinearFitOLD(basis2a, im2, verbose=False):
    pars, resid, _, _ = np.linalg.lstsq(basis2a, im2.flatten())
    fit = (pars * basis2a).sum(1).reshape(im2.shape)
    if verbose:
        print resid, np.sum((im2 - fit.reshape(im2.shape))**2)
    return pars, fit, resid

# Create the $b_i$ and $M_{ij}$ from the A&L (1998) and Becker (2012)
# papers. This was done wrong in the previous version of notebook 3
# (and above), although it gives identical results.

def doTheLinearFitAL(basis2a, im2, verbose=False):
    b = (basis2a.T * im2.flatten()).sum(1)
    M = np.dot(basis2a.T, basis2a)
    pars, resid, _, _ = np.linalg.lstsq(M, b)
    fit = (pars * basis2a).sum(1).reshape(im2.shape)
    if verbose:
        print resid, np.sum((im2 - fit.reshape(im2.shape))**2)
    return pars, fit, resid

# Also generate the matching kernel from the resulting pars.

# Look at the resulting matching kernel by multiplying the fitted
# parameters times the original basis funcs. and test that actually
# convolving it with the template gives us a good subtraction.

# Here, we'll just compute the spatial part at x,y=0,0
# (i.e. x,y=256,256 in img coords)

def getMatchingKernelAL(pars, basis, constKernelIndices, nonConstKernelIndices, spatialBasis,
                        basisScale, basisOffset=0, xcen=256, ycen=256, verbose=False):
    kbasis1 = np.vstack([b.flatten() for b in basis]).T
    kbasis1 = (kbasis1 - basisOffset) / basisScale[constKernelIndices]
    kfit1 = (pars[constKernelIndices] * kbasis1).sum(1).reshape(basis[0].shape)

    kbasis2 = np.vstack([(b[2]*b[1][xcen, ycen]).flatten() for b in spatialBasis]).T
    kbasis2 = (kbasis2 - basisOffset) / basisScale[nonConstKernelIndices]
    kfit2 = (pars[nonConstKernelIndices] * kbasis2).sum(1).reshape(basis[0].shape)

    kfit = kfit1 + kfit2
    if verbose:
        print kfit1.sum(), kfit2.sum(), kfit.sum()
    kfit /= kfit.sum()  # this is necessary if the variable source changes a lot - prevent the kernel from incorporating that change in flux
    return kfit


# Compute the "L(ZOGY)" post-conv. kernel from kfit

# Note unlike previous notebooks, here because the PSF is varying,
# we'll just use `fit2` rather than `im2-conv_im1` as the diffim,
# since `fit2` already incorporates the spatially varying PSF.
# sig1 and sig2 are the same as those input to makeFakeImages().

def computeCorrectionKernelALZC(kappa, sig1=0.2, sig2=0.2):
    def kernel_ft2(kernel):
        FFT = fft2(kernel)
        return FFT
    def post_conv_kernel_ft2(kernel, sig1=1., sig2=1.):
        kft = kernel_ft2(kernel)
        return np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))
    def post_conv_kernel2(kernel, sig1=1., sig2=1.):
        kft = post_conv_kernel_ft2(kernel, sig1, sig2)
        out = ifft2(kft)
        return out

    pck = post_conv_kernel2(kappa, sig1=sig2, sig2=sig1)
    pck = np.fft.ifftshift(pck.real)
    #print np.unravel_index(np.argmax(pck), pck.shape)

    # I think we actually need to "reverse" the PSF, as in the ZOGY (and Kaiser) papers... let's try it.
    # This is the same as taking the complex conjugate in Fourier space before FFT-ing back to real space.
    if False:
        # I still think we need to flip it in one axis (TBD: figure this out!)
        pck = pck[::-1, :]

    return pck


# Compute the (corrected) diffim's new PSF
# post_conv_psf = phi_1(k) * sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
# we'll parameterize phi_1(k) as a gaussian with sigma "psfsig1".
# im2_psf is the the psf of im2

def computeCorrectedDiffimPsfALZC(kappa, im2_psf, sig1=0.2, sig2=0.2):
    def post_conv_psf_ft2(psf, kernel, sig1=1., sig2=1.):
        # Pad psf or kernel symmetrically to make them the same size!
        if psf.shape[0] < kernel.shape[0]:
            while psf.shape[0] < kernel.shape[0]:
                psf = np.pad(psf, (1, 1), mode='constant')
        elif psf.shape[0] > kernel.shape[0]:
            while psf.shape[0] > kernel.shape[0]:
                kernel = np.pad(kernel, (1, 1), mode='constant')
        psf_ft = fft2(psf)
        kft = fft2(kernel)
        out = psf_ft * np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))
        return out
    def post_conv_psf(psf, kernel, sig1=1., sig2=1.):
        kft = post_conv_psf_ft2(psf, kernel, sig1, sig2)
        out = ifft2(kft)
        return out

    im2_psf_small = im2_psf
    # First compute the science image's (im2's) psf, subset on -16:15 coords
    if im2_psf.shape[0] > 50:
        x0im, y0im = getImageGrid(im2_psf)
        x = np.arange(-16, 16, 1)
        y = x.copy()
        x0, y0 = np.meshgrid(x, y)
        im2_psf_small = im2_psf[(x0im.max()+x.min()+1):(x0im.max()-x.min()+1),
                                (y0im.max()+y.min()+1):(y0im.max()-y.min()+1)]
    pcf = post_conv_psf(psf=im2_psf_small, kernel=kappa, sig1=sig2, sig2=sig1)
    pcf = pcf.real / pcf.real.sum()
    return pcf

def computeClippedImageStats(im, low=3, high=3):
    _, low, upp = scipy.stats.sigmaclip(im, low=low, high=high)
    tmp = im[(im > low) & (im < upp)]
    mean1 = np.nanmean(tmp)
    sig1 = np.nanstd(tmp)
    return mean1, sig1

def getImageGrid(im):
    xim = np.arange(np.int(-np.floor(im.shape[0]/2.)), np.int(np.floor(im.shape[0]/2)))
    yim = np.arange(np.int(-np.floor(im.shape[1]/2.)), np.int(np.floor(im.shape[1]/2)))
    x0im, y0im = np.meshgrid(xim, yim)
    return x0im, y0im

def performAlardLupton(im1, im2, sigGauss=None, degGauss=None, betaGauss=1,
                       spatialKernelOrder=2, spatialBackgroundOrder=2, doALZCcorrection=True,
                       sig1=None, sig2=None, verbose=False):
    x = np.arange(-16, 16, 1)
    y = x.copy()
    x0, y0 = np.meshgrid(x, y)

    basis = getALChebGaussBases(x0, y0, sigGauss=sigGauss, degGauss=degGauss,
                                betaGauss=betaGauss, verbose=verbose)
    basis2 = makeImageBases(im1, basis)
    spatialBasis, bgBasis = makeSpatialBases(im1, basis, basis2, verbose=verbose)
    basis2a, (constKernelIndices, nonConstKernelIndices, bgIndices), (basisOffset, basisScale) \
        = collectAllBases(basis2, spatialBasis, bgBasis)

    pars, fit, resid = doTheLinearFitAL(basis2a, im2)
    xcen = np.int(np.floor(im1.shape[0]/2.))
    ycen = np.int(np.floor(im1.shape[1]/2.))

    kfit = getMatchingKernelAL(pars, basis, constKernelIndices, nonConstKernelIndices,
                               spatialBasis, basisScale, basisOffset, xcen=xcen, ycen=ycen,
                               verbose=verbose)
    diffim = im2 - fit
    if doALZCcorrection:
        if sig1 is None:
            _, sig1 = computeClippedImageStats(im1)
        if sig2 is None:
            _, sig2 = computeClippedImageStats(im2)

        print sig1, sig2
        pck = computeCorrectionKernelALZC(kfit, sig1, sig2)
        pci = scipy.ndimage.filters.convolve(diffim, pck, mode='constant')
        diffim = pci

    return diffim, kfit

# Compute the ZOGY eqn. (13):
# $$
# \widehat{D} = \frac{F_r\widehat{P_r}\widehat{N} -
# F_n\widehat{P_n}\widehat{R}}{\sqrt{\sigma_n^2 F_r^2
# |\widehat{P_r}|^2 + \sigma_r^2 F_n^2 |\widehat{P_n}|^2}}
# $$
# where $D$ is the optimal difference image, $R$ and $N$ are the
# reference and "new" image, respectively, $P_r$ and $P_n$ are their
# PSFs, $F_r$ and $F_n$ are their flux-based zero-points (which we
# will set to one here), $\sigma_r^2$ and $\sigma_n^2$ are their
# variance, and $\widehat{D}$ denotes the FT of $D$.

def performZOGY(im1, im2, im1_psf, im2_psf, sig1=None, sig2=None):
    from scipy.fftpack import fft2, ifft2, ifftshift

    if sig1 is None:
        _, sig1 = computeClippedImageStats(im1)
    if sig2 is None:
        _, sig2 = computeClippedImageStats(im2)

    F_r = F_n = 1.
    R_hat = fft2(im1)
    N_hat = fft2(im2)
    P_r = im1_psf
    P_n = im2_psf
    P_r_hat = fft2(P_r)
    P_n_hat = fft2(P_n)
    d_hat_numerator = (F_r * P_r_hat * N_hat - F_n * P_n_hat * R_hat)
    d_hat_denom = np.sqrt((sig1**2 * F_r**2 * np.abs(P_r_hat)**2) + (sig2**2 * F_n**2 * np.abs(P_n_hat)**2))
    d_hat = d_hat_numerator / d_hat_denom

    d = ifft2(d_hat)
    D = ifftshift(d.real)
    return D


def computePixelCovariance(diffim, diffim2=None):
    diffim = diffim/diffim.std()
    shifted_imgs2 = None
    shifted_imgs = [
        diffim,
        np.roll(diffim, 1, 0), np.roll(diffim, -1, 0), np.roll(diffim, 1, 1), np.roll(diffim, -1, 1),
        np.roll(np.roll(diffim, 1, 0), 1, 1), np.roll(np.roll(diffim, 1, 0), -1, 1),
        np.roll(np.roll(diffim, -1, 0), 1, 1), np.roll(np.roll(diffim, -1, 0), -1, 1),
        np.roll(diffim, 2, 0), np.roll(diffim, -2, 0), np.roll(diffim, 2, 1), np.roll(diffim, -2, 1),
        np.roll(diffim, 3, 0), np.roll(diffim, -3, 0), np.roll(diffim, 3, 1), np.roll(diffim, -3, 1),
        np.roll(diffim, 4, 0), np.roll(diffim, -4, 0), np.roll(diffim, 4, 1), np.roll(diffim, -4, 1),
        np.roll(diffim, 5, 0), np.roll(diffim, -5, 0), np.roll(diffim, 5, 1), np.roll(diffim, -5, 1),
    ]
    shifted_imgs = np.vstack([i.flatten() for i in shifted_imgs])
    #out = np.corrcoef(shifted_imgs)
    if diffim2 is not None:
        shifted_imgs2 = [
            diffim2,
            np.roll(diffim2, 1, 0), np.roll(diffim2, -1, 0), np.roll(diffim2, 1, 1), np.roll(diffim2, -1, 1),
            np.roll(np.roll(diffim2, 1, 0), 1, 1), np.roll(np.roll(diffim2, 1, 0), -1, 1),
            np.roll(np.roll(diffim2, -1, 0), 1, 1), np.roll(np.roll(diffim2, -1, 0), -1, 1),
            np.roll(diffim2, 2, 0), np.roll(diffim2, -2, 0), np.roll(diffim2, 2, 1), np.roll(diffim2, -2, 1),
            np.roll(diffim2, 3, 0), np.roll(diffim2, -3, 0), np.roll(diffim2, 3, 1), np.roll(diffim2, -3, 1),
            np.roll(diffim2, 4, 0), np.roll(diffim2, -4, 0), np.roll(diffim2, 4, 1), np.roll(diffim2, -4, 1),
            np.roll(diffim2, 5, 0), np.roll(diffim2, -5, 0), np.roll(diffim2, 5, 1), np.roll(diffim2, -5, 1),
        ]
        shifted_imgs2 = np.vstack([i.flatten() for i in shifted_imgs2])
    out = np.cov(shifted_imgs, shifted_imgs2, bias=1)
    tmp2 = out.copy()
    np.fill_diagonal(tmp2, np.NaN)
    stat = np.nansum(tmp2)/np.sum(np.diag(out))  # print sum of off-diag / sum of diag
    return out, stat


# Compute ALZC correction kernel from matching kernel
# Here we use a constant kernel, just compute it for the center of the image.
def performALZCExposureCorrection(templateExposure, exposure, subtractedExposure, psfMatchingKernel, log):
    import lsst.afw.image as afwImage
    import lsst.meas.algorithms as measAlg
    import lsst.afw.math as afwMath

    spatialKernel = psfMatchingKernel
    kimg = afwImage.ImageD(spatialKernel.getDimensions())
    bbox = subtractedExposure.getBBox()
    xcen = (bbox.getBeginX() + bbox.getEndX()) / 2.
    ycen = (bbox.getBeginY() + bbox.getEndY()) / 2.
    spatialKernel.computeImage(kimg, True, xcen, ycen)
    # Compute the images' sigmas (sqrt of variance)
    sig1 = templateExposure.getMaskedImage().getVariance().getArray()
    sig2 = exposure.getMaskedImage().getVariance().getArray()
    sig1squared, _ = computeClippedImageStats(sig1)
    sig2squared, _ = computeClippedImageStats(sig2)
    sig1 = np.sqrt(sig1squared)
    sig2 = np.sqrt(sig2squared)
    corrKernel = computeCorrectionKernelALZC(kimg.getArray(), sig1=sig1, sig2=sig2)
    # Eventually, use afwMath.convolve(), but for now just use scipy.
    log.info("ALZC: Convolving.")
    pci, _ = doConvolve(subtractedExposure.getMaskedImage().getImage().getArray(),
                     corrKernel)
    subtractedExposure.getMaskedImage().getImage().getArray()[:, :] = pci
    log.info("ALZC: Finished with convolution.")

    # Compute the subtracted exposure's updated psf
    psf = subtractedExposure.getPsf().computeImage().getArray()
    psfc = computeCorrectedDiffimPsfALZC(corrKernel, psf, sig1=sig1, sig2=sig2)
    psfcI = afwImage.ImageD(subtractedExposure.getPsf().computeImage().getBBox())
    psfcI.getArray()[:, :] = psfc
    psfcK = afwMath.FixedKernel(psfcI)
    psfNew = measAlg.KernelPsf(psfcK)
    subtractedExposure.setPsf(psfNew)
    return subtractedExposure, corrKernel


def computeClippedAfwStats(im, numSigmaClip=3., numIter=3, maskIm=None):
    """! Utility function for sigma-clipped array statistics on an image or exposure.
    @param im An afw.Exposure, masked image, or image.
    @return sigma-clipped mean, std, and variance of input array
    """
    statsControl = afwMath.StatisticsControl()
    statsControl.setNumSigmaClip(numSigmaClip)
    statsControl.setNumIter(numIter)
    ignoreMaskPlanes = ["INTRP", "EDGE", "DETECTED", "SAT", "CR", "BAD", "NO_DATA", "DETECTED_NEGATIVE"]
    statsControl.setAndMask(afwImage.MaskU.getPlaneBitMask(ignoreMaskPlanes))
    if maskIm is None:
        statObj = afwMath.makeStatistics(im,
                                         afwMath.MEANCLIP | afwMath.STDEVCLIP | afwMath.VARIANCECLIP,
                                         statsControl)
    else:
        statObj = afwMath.makeStatistics(im, maskIm,
                                         afwMath.MEANCLIP | afwMath.STDEVCLIP | afwMath.VARIANCECLIP,
                                         statsControl)
    mean = statObj.getValue(afwMath.MEANCLIP)
    std = statObj.getValue(afwMath.STDEVCLIP)
    var = statObj.getValue(afwMath.VARIANCECLIP)
    return mean, std, var

def doConvolve(exposure, kernel, use_scipy=False):
    """! Convolve an Exposure with a decorrelation convolution kernel.
    @param exposure Input afw.image.Exposure to be convolved.
    @param kernel Input 2-d numpy.array to convolve the image with
    @param use_scipy Use scipy to do convolution instead of afwMath
    @return a new Exposure with the convolved pixels and the (possibly
    re-centered) kernel.

    @note We use afwMath.convolve() but keep scipy.convolve for debugging.
    @note We re-center the kernel if necessary and return the possibly re-centered kernel
    """
    def _fixEvenKernel(kernel):
        """! Take a kernel with even dimensions and make them odd, centered correctly.
        @param kernel a numpy.array
        @return a fixed kernel numpy.array
        """
        # Make sure the peak (close to a delta-function) is in the center!
        maxloc = np.unravel_index(np.argmax(kernel), kernel.shape)
        out = np.roll(kernel, kernel.shape[0]//2 - maxloc[0], axis=0)
        out = np.roll(out, out.shape[1]//2 - maxloc[1], axis=1)
        # Make sure it is odd-dimensioned by trimming it.
        if (out.shape[0] % 2) == 0:
            maxloc = np.unravel_index(np.argmax(out), out.shape)
            if out.shape[0] - maxloc[0] > maxloc[0]:
                out = out[:-1, :]
            else:
                out = out[1:, :]
            if out.shape[1] - maxloc[1] > maxloc[1]:
                out = out[:, :-1]
            else:
                out = out[:, 1:]
        return out

    outExp = kern = None
    fkernel = _fixEvenKernel(kernel)
    if use_scipy:
        from scipy.ndimage.filters import convolve
        pci = convolve(exposure.getMaskedImage().getImage().getArray(),
                       fkernel, mode='constant', cval=np.nan)
        outExp = exposure.clone()
        outExp.getMaskedImage().getImage().getArray()[:, :] = pci
        kern = fkernel

    else:
        kernelImg = afwImage.ImageD(fkernel.shape[0], fkernel.shape[1])
        kernelImg.getArray()[:, :] = fkernel
        kern = afwMath.FixedKernel(kernelImg)
        maxloc = np.unravel_index(np.argmax(fkernel), fkernel.shape)
        kern.setCtrX(maxloc[0])
        kern.setCtrY(maxloc[1])
        outExp = exposure.clone()  # Do this to keep WCS, PSF, masks, etc.
        convCntrl = afwMath.ConvolutionControl(False, True, 0)
        afwMath.convolve(outExp.getMaskedImage(), exposure.getMaskedImage(), kern, convCntrl)

    return outExp, kern

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
