import numpy as np
import scipy

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

    tmp = np.sqrt(len(images))
    nrows_ncols = (np.int(np.floor(tmp)), np.int(np.ceil(len(images)/np.int(np.floor(tmp))))) if nrows_ncols is None else nrows_ncols
    size = (nrows_ncols[1]*imScale, nrows_ncols[0]*imScale)
    fig = plt.figure(1, size)
    igrid = ImageGrid(fig, 111,  # similar to subplot(111)
                      nrows_ncols=nrows_ncols, direction='row',  # creates 2x2 grid of axes
                      axes_pad=0.1,  # pad between axes in inch.
                      label_mode="L",  # share_all=True,
                      cbar_location="right", cbar_mode="single", cbar_size='7%')
    for i in range(len(images)):
        ii = images[i]
        if cbar and clim is not None:
            ii = np.clip(ii, clim[0], clim[1])
        im = igrid[i].imshow(ii, origin='lower', interpolation=interpolation, cmap=cmap,
                        extent=extent, clim=clim)
        if cbar:
            igrid[i].cax.colorbar(im)
        if titles is not None:  # assume titles is an array or tuple of same length as images.
            t = add_inner_title(igrid[i], titles[i], loc=2)
            t.patch.set_ec("none")
            t.patch.set_alpha(0.5)
    return igrid

def gaussian2d(grid, m=None, s=None):
    import scipy.stats
    ## see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

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
    from scipy.fftpack import fft2, ifft2, fftfreq, fftshift

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
        pck = pck[::-1,:]

    return pck


# Compute the (corrected) diffim's new PSF
# post_conv_psf = phi_1(k) * sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
# we'll parameterize phi_1(k) as a gaussian with sigma "psfsig1".
# im2_psf is the the psf of im2

def computeCorrectedDiffimPsfALZC(kappa, im2_psf, sig1=0.2, sig2=0.2):
    from scipy.fftpack import fft2, ifft2, fftfreq, fftshift

    def kernel_ft2(kernel):
        FFT = fft2(kernel)
        return FFT
    def post_conv_psf_ft2(psf, kernel, sig1=1., sig2=1.):
        psf_ft = kernel_ft2(psf)
        kft = kernel_ft2(kernel)
        out = psf_ft * np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))
        return out
    def post_conv_psf(psf, kernel, sig1=1., sig2=1.):
        kft = post_conv_psf_ft2(psf, kernel, sig1, sig2)
        out = ifft2(kft)
        return out

    # First compute the science image's (im2's) psf, subset on -16:15 coords
    x0im, y0im = getImageGrid(im2_psf)
    x = np.arange(-16, 16, 1)
    y = x.copy()
    x0, y0 = np.meshgrid(x, y)

    im2_psf_small = im2_psf[(x0im.max()+x.min()+1):(x0im.max()-x.min()+1),
                            (y0im.max()+y.min()+1):(y0im.max()-y.min()+1)]
    print im2_psf_small.shape
    pcf = post_conv_psf(psf=im2_psf_small, kernel=kappa, sig1=sig2, sig2=sig1)
    pcf = pcf.real / pcf.real.sum()
    return pcf

def computeClippedImageStats(im, low=3, high=3):
    _, low, upp = scipy.stats.sigmaclip(im, low=low, high=high)
    tmp = im[(im>low) & (im<upp)]
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

    #xim = np.arange(np.int(-np.floor(im1.shape[0]/2.)), np.int(np.floor(im1.shape[0]/2)))
    #yim = np.arange(np.int(-np.floor(im1.shape[1]/2.)), np.int(np.floor(im1.shape[1]/2)))
    x0im, y0im = getImageGrid(im1) #np.meshgrid(xim, yim)
    F_r = F_n = 1.
    R_hat = fft2(im1)
    N_hat = fft2(im2)
    P_r = im1_psf #singleGaussian2d(x0im, y0im, 0, 0, psf1, psf1)
    P_n = im2_psf #singleGaussian2d(x0im, y0im, 0, 0, psf2, psf2*1.5)
    P_r_hat = fft2(P_r)
    P_n_hat = fft2(P_n)
    d_hat_numerator = (F_r * P_r_hat * N_hat - F_n * P_n_hat * R_hat)
    d_hat_denom = np.sqrt((sig1**2 * F_r**2 * np.abs(P_r_hat)**2) + (sig2**2 * F_n**2 * np.abs(P_n_hat)**2))
    d_hat = d_hat_numerator / d_hat_denom

    d = ifft2(d_hat)
    D = ifftshift(d.real)
    return D


def computePixelCovariance(diffim):
    diffim = diffim/diffim.std()
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
    out = np.corrcoef(shifted_imgs)
    out = np.cov(shifted_imgs, bias=1)
    tmp2 = out.copy()
    np.fill_diagonal(tmp2, np.NaN)
    print np.nansum(tmp2)/np.sum(np.diag(out))  ## print sum of off-diag / sum of diag
    #np.fill_diagonal(out, np.NaN)
    #inds = np.argsort(out[0,:])[::-1]  # sort (decreasing) by cov with center pixel
    #np.fill_diagonal(out, 0)
    #inds = np.argsort(out.sum(0))[::-1]
    return out#[inds, :][:, inds]
