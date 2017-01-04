import numpy as np
from numpy.polynomial.chebyshev import chebval2d
import scipy.ndimage.filters

from . import psf
from . import utils

# Okay, here we start the A&L basis functions...
# Update: it looks like the actual code in the stack uses chebyshev1 polynomials!
# Note these are essentially the same but with different scale factors.

# Here beta is a rescale factor but this is NOT what it is really used for.
# Can be used to rescale so that sigGauss[1] = sqrt(sigmaPsf_I^2 - sigmaPsf_T^2)
def chebGauss2d(x, y, m=None, s=None, ord=[0, 0], beta=1., verbose=False):
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
    ga = psf.singleGaussian2d(x, y, 0, 0, s[0]/beta, s[1]/beta)
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
    basis = [chebGauss2d(x0, y0, m=[0,0], s=[sig,sig], ord=[inds[i][0][ind], inds[i][1][ind]],
                         beta=betaGauss, verbose=verbose) for i,sig in enumerate(sigGauss) for
             ind in range(len(inds[i][0]))]
    return basis

# Convolve im1 (template) with the basis functions, and make these the *new* bases.
# Input 'basis' is the output of getALChebGaussBases().

def makeImageBases(im1, basis):
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
    x0im, y0im = utils.getImageGrid(im1)  # np.meshgrid(xim, yim)

    # Note the ordering of the loop is important! Make the basis2 the last one so the first set of values
    # that are returned are all of the original (basis2) unmodified bases.
    # Store "spatialBasis" which is the kernel basis and the spatial basis separated so we can recompute the
    # final kernel at the end. Include in index 2 the "original" kernel basis as well.
    if spatialKernelOrder > 0:
        spatialBasis = [[basis2[bi], cheb2d(x0im, y0im, ord=[spatialInds[0][i], spatialInds[1][i]], verbose=False),
                         basis[bi]] for i in range(1,len(spatialInds[0])) for bi in range(len(basis2))]
        # basis2m = [b * cheb2d(x0im, y0im, ord=[spatialInds[0][i], spatialInds[1][i]], verbose=False) for
        # i in range(1,len(spatialInds[0])) for b in basis2]

    spatialBgInds = get_valid_inds(spatialBackgroundOrder)
    if verbose:
        print spatialBgInds

    # Then make the spatial background part
    if spatialBackgroundOrder > 0:
        bgBasis = [cheb2d(x0im, y0im, ord=[spatialBgInds[0][i], spatialBgInds[1][i]], verbose=False) for
                   i in range(len(spatialBgInds[0]))]

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
    # this is necessary if the source changes a lot - prevent the kernel from incorporating that change in flux
    kfit /= kfit.sum()
    return kfit


# Here, im2 is science, im1 is template
def performAlardLupton(im1, im2, sigGauss=None, degGauss=None, betaGauss=1, kernelSize=25,
                       spatialKernelOrder=2, spatialBackgroundOrder=2, doALZCcorrection=True,
                       preConvKernel=None, sig1=None, sig2=None, im2Psf=None, verbose=False):
    x = np.arange(-kernelSize+1, kernelSize, 1)
    y = x.copy()
    x0, y0 = np.meshgrid(x, y)

    im2_orig = im2
    if preConvKernel is not None:
        im2 = scipy.ndimage.filters.convolve(im2, preConvKernel, mode='constant')

    basis = getALChebGaussBases(x0, y0, sigGauss=sigGauss, degGauss=degGauss,
                                betaGauss=betaGauss, verbose=verbose)
    basis2 = makeImageBases(im1, basis)
    spatialBasis, bgBasis = makeSpatialBases(im1, basis, basis2, verbose=verbose)
    basis2a, (constKernelIndices, nonConstKernelIndices, bgIndices), (basisOffset, basisScale) \
        = collectAllBases(basis2, spatialBasis, bgBasis)
    del bgBasis
    del basis2

    pars, fit, resid = doTheLinearFitAL(basis2a, im2)
    del basis2a
    xcen = np.int(np.floor(im1.shape[0]/2.))
    ycen = np.int(np.floor(im1.shape[1]/2.))

    kfit = getMatchingKernelAL(pars, basis, constKernelIndices, nonConstKernelIndices,
                               spatialBasis, basisScale, basisOffset, xcen=xcen, ycen=ycen,
                               verbose=verbose)
    del basis
    del spatialBasis
    diffim = im2 - fit
    psf = im2Psf
    if doALZCcorrection:
        if sig1 is None:
            _, sig1, _, _ = computeClippedImageStats(im1)
        if sig2 is None:
            _, sig2, _, _ = computeClippedImageStats(im2)

        pck = computeDecorrelationKernel(kfit, sig1**2, sig2**2, preConvKernel=preConvKernel)
        pci = scipy.ndimage.filters.convolve(diffim, pck, mode='constant')
        if im2Psf is not None:
            psf = computeCorrectedDiffimPsf(kfit, im2Psf, svar=sig1**2, tvar=sig2**2)
        diffim = pci

    return diffim, psf, kfit


# Compute the "ALZC" post-conv. kernel from kfit

# Note unlike previous notebooks, here because the PSF is varying,
# we'll just use `fit2` rather than `im2-conv_im1` as the diffim,
# since `fit2` already incorporates the spatially varying PSF.
# sig1 and sig2 are the same as those input to makeFakeImages().

# def computeCorrectionKernelALZC(kappa, sig1=0.2, sig2=0.2):
#     def kernel_ft2(kernel):
#         FFT = fft2(kernel)
#         return FFT
#     def post_conv_kernel_ft2(kernel, sig1=1., sig2=1.):
#         kft = kernel_ft2(kernel)
#         return np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))
#     def post_conv_kernel2(kernel, sig1=1., sig2=1.):
#         kft = post_conv_kernel_ft2(kernel, sig1, sig2)
#         out = ifft2(kft)
#         return out

#     pck = post_conv_kernel2(kappa, sig1=sig2, sig2=sig1)
#     pck = np.fft.ifftshift(pck.real)
#     #print np.unravel_index(np.argmax(pck), pck.shape)

#     # I think we actually need to "reverse" the PSF, as in the ZOGY (and Kaiser) papers... let's try it.
#     # This is the same as taking the complex conjugate in Fourier space before FFT-ing back to real space.
#     if False:
#         # I still think we need to flip it in one axis (TBD: figure this out!)
#         pck = pck[::-1, :]

#     return pck


# Compute the (corrected) diffim's new PSF
# post_conv_psf = phi_1(k) * sym.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kappa_ft(k)**2))
# we'll parameterize phi_1(k) as a gaussian with sigma "psfsig1".
# im2_psf is the the psf of im2

# def computeCorrectedDiffimPsfALZC(kappa, im2_psf, sig1=0.2, sig2=0.2):
#     def post_conv_psf_ft2(psf, kernel, sig1=1., sig2=1.):
#         # Pad psf or kernel symmetrically to make them the same size!
#         if psf.shape[0] < kernel.shape[0]:
#             while psf.shape[0] < kernel.shape[0]:
#                 psf = np.pad(psf, (1, 1), mode='constant')
#         elif psf.shape[0] > kernel.shape[0]:
#             while psf.shape[0] > kernel.shape[0]:
#                 kernel = np.pad(kernel, (1, 1), mode='constant')
#         psf_ft = fft2(psf)
#         kft = fft2(kernel)
#         out = psf_ft * np.sqrt((sig1**2 + sig2**2) / (sig1**2 + sig2**2 * kft**2))
#         return out
#     def post_conv_psf(psf, kernel, sig1=1., sig2=1.):
#         kft = post_conv_psf_ft2(psf, kernel, sig1, sig2)
#         out = ifft2(kft)
#         return out

#     im2_psf_small = im2_psf
#     # First compute the science image's (im2's) psf, subset on -16:15 coords
#     if im2_psf.shape[0] > 50:
#         x0im, y0im = getImageGrid(im2_psf)
#         x = np.arange(-16, 16, 1)
#         y = x.copy()
#         x0, y0 = np.meshgrid(x, y)
#         im2_psf_small = im2_psf[(x0im.max()+x.min()+1):(x0im.max()-x.min()+1),
#                                 (y0im.max()+y.min()+1):(y0im.max()-y.min()+1)]
#     pcf = post_conv_psf(psf=im2_psf_small, kernel=kappa, sig1=sig2, sig2=sig1)
#     pcf = pcf.real / pcf.real.sum()
#     return pcf
