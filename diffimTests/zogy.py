import numpy as np
import scipy.signal

from .utils import computeClippedImageStats, memoize

# Compute the ZOGY eqn. (13):
# $$
# \widehat{D} = \frac{Fr\widehat{Pr}\widehat{N} -
# F_n\widehat{Pn}\widehat{R}}{\sqrt{\sigma_n^2 Fr^2
# |\widehat{Pr}|^2 + \sigma_r^2 F_n^2 |\widehat{Pn}|^2}}
# $$
# where $D$ is the optimal difference image, $R$ and $N$ are the
# reference and "new" image, respectively, $Pr$ and $P_n$ are their
# PSFs, $Fr$ and $Fn$ are their flux-based zero-points (which we
# will set to one here), $\sigma_r^2$ and $\sigma_n^2$ are their
# variance, and $\widehat{D}$ denotes the FT of $D$.

# In all functions, im1 is R (reference, or template) and im2 is N (new, or science)

@memoize  # Don't want this in production! Find another way to store the results of this func
def computeZogyPrereqs(im1, im2, im1_psf, im2_psf, sig1=None, sig2=None, Fr=1., Fn=1., padSize=0):
    if sig1 is None and im1 is not None:
        _, sig1, _, _ = computeClippedImageStats(im1)
    if sig2 is None and im2 is not None:
        _, sig2, _, _ = computeClippedImageStats(im2)

    psf1 = im1_psf
    psf2 = im2_psf
    if padSize > 0:
        padSize0 = padSize #im1.shape[0]//2 - im1_psf.shape[0]//2 # Need to pad the PSF to remove windowing artifacts
        padSize1 = padSize #im1.shape[1]//2 - im1_psf.shape[1]//2 # The bigger the padSize the better, but slower.
        psf1 = np.pad(im1_psf, ((padSize0, padSize0), (padSize1, padSize1)), mode='constant',
                      constant_values=0)
        psf1 *= im1_psf.mean() / psf1.mean()
        psf2 = np.pad(im2_psf, ((padSize0, padSize0), (padSize1, padSize1)), mode='constant',
                      constant_values=0)
        psf2 *= im2_psf.mean() / psf2.mean()

    Pr = psf1 #im1_psf
    Pn = psf2 #im2_psf
    sigR = sig1
    sigN = sig2
    Pr_hat = np.fft.fft2(Pr)
    Pn_hat = np.fft.fft2(Pn)
    denom = np.sqrt((sigN**2 * Fr**2 * np.abs(Pr_hat)**2) + (sigR**2 * Fn**2 * np.abs(Pn_hat)**2))
    Fd = Fr*Fn / np.sqrt(sigN**2 * Fr**2 + sigR**2 * Fn**2)

    output = {#'sigR': sigR, 'sigN': sigN,
              'Pr_hat': Pr_hat, 'Pn_hat': Pn_hat,
              'denom': denom,
              'Pr': Pr, 'Pn': Pn,
              'Fd': Fd}
    return output


# In all functions, im1 is R (reference, or template) and im2 is N (new, or science)
def performZogy(im1, im2, im1_var, im2_var, im1_psf=None, im2_psf=None, sig1=None, sig2=None,
                Fr=1., Fn=1.):
    # Do all in fourier space (needs image-sized PSFs)
    padSize0 = im1.shape[0]//2 - im1_psf.shape[0]//2
    padSize1 = im1.shape[1]//2 - im1_psf.shape[1]//2
    psf1, psf2 = im1_psf, im2_psf
    # Hastily assume the image is even-sized and the psf is odd...
    if padSize0 > 0 or padSize1 > 0:
        if padSize0 < 0:
            padSize0 = 0
        if padSize1 < 0:
            padSize1 = 0
        psf1 = np.pad(im1_psf, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
                      constant_values=0)
        psf2 = np.pad(im2_psf, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
                      constant_values=0)

    prereqs = computeZogyPrereqs(im1, im2, psf1, psf2,
                                 sig1, sig2, Fr, Fn, padSize=0)
    Pr_hat, Pn_hat, denom, Fd = (prereqs[key] for key in
                                 ['Pr_hat', 'Pn_hat', 'denom', 'Fd'])

    # First do the image
    R_hat = np.fft.fft2(im1)
    N_hat = np.fft.fft2(im2)
    D_hat = (Fr * Pr_hat * N_hat - Fn * Pn_hat * R_hat)
    D_hat /= denom

    D = np.fft.ifft2(D_hat)
    D = np.fft.ifftshift(D.real) / Fd
    #D *= np.sqrt(sigR**2. + sigN**2.)  # Set to same scale as A&L

    # Do the exact same thing to the var images, except add them
    R_hat = np.fft.fft2(im1_var)
    N_hat = np.fft.fft2(im2_var)
    d_hat = (Fr * Pr_hat * N_hat + Fn * Pn_hat * R_hat)
    d_hat /= denom

    d = np.fft.ifft2(d_hat)
    D_var = np.fft.ifftshift(d.real) / Fd
    #D_var *= np.sqrt(sigR**2. + sigN**2.)  # Set to same scale as A&L

    return D, D_var


# In all functions, im1 is R (reference, or template) and im2 is N (new, or science)
def performZogyImageSpace(im1, im2, im1_var, im2_var, im1_psf=None, im2_psf=None,
                          sig1=None, sig2=None, Fr=1., Fn=1., padSize=7):
    prereqs = computeZogyPrereqs(im1, im2, im1_psf, im2_psf,
                                 sig1, sig2, Fr, Fn, padSize=padSize)
    Pr_hat, Pn_hat, denom, padded_psf1, padded_psf2, Fd = (prereqs[key] for key in
        ['Pr_hat', 'Pn_hat', 'denom', 'Pr', 'Pn', 'Fd'])

    delta = 0. #.1
    Kr_hat = (Pr_hat + delta) / (denom + delta)
    Kn_hat = (Pn_hat + delta) / (denom + delta)
    Kr = np.fft.ifft2(Kr_hat).real
    Kn = np.fft.ifft2(Kn_hat).real

    if padSize > 0:
        ps = padSize #// 2
        Kn = Kn[ps:-ps, ps:-ps]
        Kr = Kr[ps:-ps, ps:-ps]

    # Note these are reverse-labelled, this is CORRECT!
    im1c = scipy.ndimage.filters.convolve(im1, Kn, mode='constant', cval=np.nan)
    im2c = scipy.ndimage.filters.convolve(im2, Kr, mode='constant', cval=np.nan)
    D = (im2c - im1c) / Fd
    #D *= np.sqrt(sigR**2. + sigN**2.)  # Set to same scale as A&L

    # Do the same convolutions to the variance images
    im1c = scipy.ndimage.filters.convolve(im1_var, Kn, mode='constant', cval=np.nan)
    im2c = scipy.ndimage.filters.convolve(im2_var, Kr, mode='constant', cval=np.nan)
    D_var = (im2c + im1c) / Fd
    #D_var *= np.sqrt(sigR**2. + sigN**2.)  # Set to same scale as A&L

    return D, D_var


## Also compute the diffim's PSF (eq. 14)
def computeZogyDiffimPsf(im1, im2, im1_psf, im2_psf, sig1=None, sig2=None,
                         Fr=1., Fn=1., padSize=0, keepFourier=False):
    prereqs = computeZogyPrereqs(im1, im2, im1_psf, im2_psf,
                                 sig1, sig2, Fr, Fn, padSize=padSize)
    Pr_hat, Pn_hat, denom, Fd = (prereqs[key] for key in ['Pr_hat', 'Pn_hat', 'denom', 'Fd'])

    Pd_hat_numerator = (Fr * Fn * Pr_hat * Pn_hat)
    Pd_hat = Pd_hat_numerator / (Fd * denom)

    if keepFourier:
        return Pd_hat

    Pd = np.fft.ifft2(Pd_hat)
    Pd = np.fft.ifftshift(Pd).real

    return Pd


def computeZogy_Scorr(D, im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                      sig1=None, sig2=None, Fr=1., Fn=1., xVarAst=0., yVarAst=0.,
                      padSize=7, inImageSpace=False):
    if inImageSpace:
        if D is None:
            D, _ = performZogyImageSpace(im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                                         sig1=sig1, sig2=sig2, Fr=Fr, Fn=Fn, padSize=padSize)
        padSize = 0
    else:
        # padSize = 0
        # padSize0 = im1.shape[0]//2 - im1_psf.shape[0]//2
        # padSize1 = im1.shape[1]//2 - im1_psf.shape[1]//2
        # # Hastily assume the image is even-sized and the psf is odd...
        # psf1 = np.pad(im1_psf, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
        #               constant_values=0)
        # psf2 = np.pad(im2_psf, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
        #               constant_values=0)
        if D is None:
            # D, _ = performZogy(im1, im2, im1_var, im2_var, psf1, psf2, sig1=sig1, sig2=sig2,
            #                    Fr=Fr, Fn=Fn)
            D, _ = performZogy(im1, im2, im1_var, im2_var, im1_psf, im2_psf, sig1=sig1, sig2=sig2,
                               Fr=Fr, Fn=Fn)
        padSize = 0

    prereqs = computeZogyPrereqs(im1, im2, im1_psf, im2_psf,
                                 sig1, sig2, Fr, Fn, padSize=padSize)
    Pr_hat, Pn_hat, denom, Fd = (prereqs[key] for key in ['Pr_hat', 'Pn_hat', 'denom', 'Fd'])

    Pd = computeZogyDiffimPsf(im1, im2, im1_psf, im2_psf, sig1, sig2, Fr, Fn)
    Pd_bar = np.fliplr(np.flipud(Pd))
    S = scipy.ndimage.filters.convolve(D, Pd_bar, mode='constant', cval=np.nan)
    S *= Fd

    # Adjust the variance planes of the two images to contribute to the final detection
    # (eq's 26-29).
    kr_hat = Fr * Fn**2. * np.conj(Pr_hat) * np.abs(Pn_hat)**2. / denom**2.
    kn_hat = Fn * Fr**2. * np.conj(Pn_hat) * np.abs(Pr_hat)**2. / denom**2.

    kr = np.fft.ifft2(kr_hat).real
    kr = np.roll(np.roll(kr, -1, 0), -1, 1)
    kn = np.fft.ifft2(kn_hat).real
    kn = np.roll(np.roll(kn, -1, 0), -1, 1)
    var1c = scipy.ndimage.filters.convolve(im1_var, kr**2., mode='constant', cval=np.nan)
    var2c = scipy.ndimage.filters.convolve(im2_var, kn**2., mode='constant', cval=np.nan)

    fGradR = fGradN = 0.
    if xVarAst + yVarAst > 0:  # Do the astrometric variance correction
        S_R = scipy.ndimage.filters.convolve(im1, kr, mode='constant', cval=np.nan)
        gradRx, gradRy = np.gradient(S_R)
        fGradR = xVarAst * gradRx**2. + yVarAst * gradRy**2.
        S_N = scipy.ndimage.filters.convolve(im2, kn, mode='constant', cval=np.nan)
        gradNx, gradNy = np.gradient(S_N)
        fGradN = xVarAst * gradNx**2. + yVarAst * gradNy**2.

    S_var = np.sqrt(var1c + var2c + fGradR + fGradN)
    S_var /= Fd
    return S, S_var, Pd, Fd


class Zogy(object):
    def __init__(self, im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                 sig1=None, sig2=None, Fr=1., Fn=1., padSize=7):
        self.im1, self.im2 = im1, im2
        self.im1_var, self.im2_var = im1_var, im2_var
        self.im1_psf, self.im2_psf = im1_psf, im2_psf
        self.sig1, self.sig2 = sig1, sig2
        self.Fr, self.Fn = Fr, Fn
        self.padSize = padSize

        if self.sig1 is None:
            _, self.sig1, _, _ = computeClippedImageStats(im1)
        if self.sig2 is None:
            _, self.sig2, _, _ = computeClippedImageStats(im2)

    def _zogyImageSpace(self):
        D, D_var = performZogyImageSpace(self.im1, self.im2, self.im1_var, self.im2_var,
                                         im1_psf=self.im1_psf, im2_psf=self.im2_psf,
                                         sig1=self.sig1, sig2=self.sig2,
                                         Fr=self.Fr, Fn=self.Fn, padSize=7)
        Pd = computeZogyDiffimPsf(self.im1, self.im2, self.im1_psf, self.im2_psf,
                                  self.sig1, self.sig2, self.Fr, self.Fn)
        return D, D_var, Pd

    def _zogyPure(self):  # non-image-space version
        D, D_var = performZogy(self.im1, self.im2, self.im1_var, self.im2_var,
                               im1_psf=self.im1_psf, im2_psf=self.im2_psf,
                               sig1=self.sig1, sig2=self.sig2,
                               Fr=self.Fr, Fn=self.Fn)
        Pd = computeZogyDiffimPsf(self.im1, self.im2, self.im1_psf, self.im2_psf,
                                  self.sig1, self.sig2, self.Fr, self.Fn)
        return D, D_var, Pd

    def _zogyScorr(self, D=None, varAst=[0., 0.]):
        S, S_var, Pd, Fd = computeZogy_Scorr(D, self.im1, self.im2, self.im1_var, self.im2_var,
            im1_psf=self.im1_psf, im2_psf=self.im2_psf, sig1=self.sig1, sig2=self.sig2,
            Fr=self.Fr, Fn=self.Fn, xVarAst=varAst[0], yVarAst=varAst[1], # these are already variances.
            padSize=0)
        return S, S_var, Pd

    def doZogy(self, inImageSpace=False, computeScorr=False):
        if inImageSpace:
            D, D_var, Pd = self._zogyImageSpace()
        else:
            D, D_var, Pd = self._zogyPure()

        if computeScorr:
            S, S_var, Pd = self._zogyScorr(D, varAst=[0., 0.])
            return D, D_var, S, S_var, Pd

        return D, D_var, Pd

