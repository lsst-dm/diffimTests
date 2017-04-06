import numpy as np
import scipy.signal

from .utils import computeClippedImageStats
from .exposure import Exposure

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

def padPsfToSize(psf, size):
    padSize0 = size[0]  # im.shape[0]//2 - psf.shape[0]//2
    padSize1 = size[1]  # im.shape[1]//2 - psf.shape[1]//2
    psf1 = psf
    # Hastily assume the image is even-sized and the psf is odd...
    if padSize0 > 0 or padSize1 > 0:
        if padSize0 < 0:
            padSize0 = 0
        if padSize1 < 0:
            padSize1 = 0
        psf1 = np.pad(psf, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
                      constant_values=0)
        #psf1 *= im1_psf.mean() / psf1.mean()
    return psf1


def padPsfToImageSize(im, psf):
    # Do all in fourier space (needs image-sized PSFs)
    return padPsfToSize(psf, (im.shape[0]//2 - psf.shape[0]//2, im.shape[1]//2 - psf.shape[1]//2))


def computeZogyPrereqs(im1, im2, im1_psf, im2_psf, sig1=None, sig2=None, Fr=1., Fn=1., padSize=0):
    if sig1 is None and im1 is not None:
        _, sig1, _, _ = computeClippedImageStats(im1)
    if sig2 is None and im2 is not None:
        _, sig2, _, _ = computeClippedImageStats(im2)

    psf1 = im1_psf
    psf2 = im2_psf
    if padSize > 0:
        psf1 = padPsfToSize(psf1, (padSize, padSize))
        psf2 = padPsfToSize(psf2, (padSize, padSize))

    Pr = psf1
    Pn = psf2
    sigR = sig1
    sigN = sig2
    Pr_hat = np.fft.fft2(Pr)
    #Pr_hat2 = np.conj(Pr_hat) * Pr_hat
    Pn_hat = np.fft.fft2(Pn)
    #Pn_hat2 = np.conj(Pn_hat) * Pn_hat
    denom = np.sqrt((sigN**2 * Fr**2 * np.abs(Pr_hat)**2) + (sigR**2 * Fn**2 * np.abs(Pn_hat)**2))
    #denom = np.sqrt((sigN**2 * Fr**2 * Pr_hat2) + (sigR**2 * Fn**2 * Pn_hat2))
    Fd = Fr*Fn / np.sqrt(sigN**2 * Fr**2 + sigR**2 * Fn**2)

    output = {#'sigR': sigR, 'sigN': sigN,
              'Pr_hat': Pr_hat, 'Pn_hat': Pn_hat,
              'denom': denom,
              'Pr': Pr, 'Pn': Pn,
              'Fd': Fd}
    return output


# In all functions, im1 is R (reference, or template) and im2 is N (new, or science)
def computeZogyFourierSpace(im1, im2, im1_var, im2_var, im1_psf=None, im2_psf=None,
                            sig1=None, sig2=None, Fr=1., Fn=1.):
    # Do all in fourier space (needs image-sized PSFs)
    psf1 = padPsfToImageSize(im1, im1_psf)
    psf2 = padPsfToImageSize(im2, im2_psf)

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

    # Do the exact same thing to the var images, except add them
    R_hat = np.fft.fft2(im1_var)
    N_hat = np.fft.fft2(im2_var)
    D_hat = (Fr * Pr_hat * N_hat + Fn * Pn_hat * R_hat)
    D_hat /= denom

    D_var = np.fft.ifft2(D_hat)
    D_var = np.fft.ifftshift(D_var.real) / Fd

    return D, D_var


# In all functions, im1 is R (reference, or template) and im2 is N (new, or science)
def computeZogyImageSpace(im1, im2, im1_var, im2_var, im1_psf=None, im2_psf=None,
                          sig1=None, sig2=None, Fr=1., Fn=1., padSize=7):
    prereqs = computeZogyPrereqs(im1, im2, im1_psf, im2_psf,
                                 sig1, sig2, Fr, Fn, padSize=padSize)
    Pr_hat, Pn_hat, denom, padded_psf1, padded_psf2, Fd = (prereqs[key] for key in
        ['Pr_hat', 'Pn_hat', 'denom', 'Pr', 'Pn', 'Fd'])

    delta = 0.  #1
    Kr_hat = (Pr_hat + delta) / (denom + delta)
    Kn_hat = (Pn_hat + delta) / (denom + delta)
    Kr = np.fft.ifft2(Kr_hat).real
    Kn = np.fft.ifft2(Kn_hat).real

    if padSize > 0:
        ps = padSize #// 2
        Kn = Kn[ps:-ps, ps:-ps]
        Kr = Kr[ps:-ps, ps:-ps]

    # if True and im1_psf.shape[0] == 41:   # it's a measured psf (hack!) This *really* helps for measured psfs.
    #     # filter the wings of Kn, Kr (see notebook #15)
    #     Knsum = Kn.mean()
    #     Kn[0:10, :] = Kn[:, 0:10] = Kn[31:41, :] = Kn[:, 31:41] = 0
    #     Kn *= Knsum / Kn.mean()
    #     Krsum = Kr.mean()
    #     Kr[0:10, :] = Kr[:, 0:10] = Kr[31:41, :] = Kr[:, 31:41] = 0
    #     Kr *= Krsum / Kr.mean()


    # Note these are reverse-labelled, this is CORRECT!
    im1c = scipy.ndimage.filters.convolve(im1, Kn, mode='constant', cval=np.nan)
    im2c = scipy.ndimage.filters.convolve(im2, Kr, mode='constant', cval=np.nan)
    D = (im2c - im1c) / Fd

    # Do the same convolutions to the variance images
    im1c = scipy.ndimage.filters.convolve(im1_var, Kn, mode='constant', cval=np.nan)
    im2c = scipy.ndimage.filters.convolve(im2_var, Kr, mode='constant', cval=np.nan)
    D_var = (im2c + im1c) / Fd

    return D, D_var


def computeZogy(im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                sig1=None, sig2=None, Fr=1., Fn=1., padSize=0, inImageSpace=True):
    if inImageSpace:
        return computeZogyImageSpace(im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                                     sig1=sig1, sig2=sig2, Fr=1., Fn=1., padSize=padSize)
    else:
        return computeZogyFourierSpace(im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                                       sig1=sig1, sig2=sig2, Fr=Fr, Fn=Fn)


## Compute the diffim's PSF (eq. 14)
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


def computeZogyScorrFourierSpace(im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                                 sig1=None, sig2=None, Fr=1., Fn=1., xVarAst=0., yVarAst=0.):
    # Do all in fourier space (needs image-sized PSFs)
    psf1 = padPsfToImageSize(im1, im1_psf)
    psf2 = padPsfToImageSize(im2, im2_psf)

    prereqs = computeZogyPrereqs(im1, im2, psf1, psf2,
                                 sig1, sig2, Fr, Fn, padSize=0)
    Pr_hat, Pn_hat, denom, Fd = (prereqs[key] for key in ['Pr_hat', 'Pn_hat', 'denom', 'Fd'])

    # Compute D_hat here (don't need D then, for speed)
    R_hat = np.fft.fft2(im1)
    N_hat = np.fft.fft2(im2)
    D_hat = (Fr * Pr_hat * N_hat - Fn * Pn_hat * R_hat)
    D_hat /= denom

    Pd_hat = computeZogyDiffimPsf(im1, im2, psf1, psf2, sig1, sig2, Fr, Fn,
                                  padSize=0, keepFourier=True)
    Pd_bar = np.conj(Pd_hat)
    S = np.fft.ifft2(D_hat * Pd_bar)

    # Adjust the variance planes of the two images to contribute to the final detection
    # (eq's 26-29).
    #Pn_hat2 = np.conj(Pn_hat) * Pn_hat
    #Kr_hat = Fr * Fn**2. * np.conj(Pr_hat) * Pn_hat2 / denom**2.
    Kr_hat = Fr * Fn**2. * np.conj(Pr_hat) * np.abs(Pn_hat)**2. / denom**2.
    #Pr_hat2 = np.conj(Pr_hat) * Pr_hat
    #Kn_hat = Fn * Fr**2. * np.conj(Pn_hat) * Pr_hat2 / denom**2.
    Kn_hat = Fn * Fr**2. * np.conj(Pn_hat) * np.abs(Pr_hat)**2. / denom**2.

    Kr_hat2 = np.fft.fft2(np.fft.ifft2(Kr_hat)**2)
    Kn_hat2 = np.fft.fft2(np.fft.ifft2(Kn_hat)**2)
    var1c_hat = Kr_hat2 * np.fft.fft2(im1_var)
    var2c_hat = Kn_hat2 * np.fft.fft2(im2_var)

    fGradR = fGradN = 0.
    if xVarAst + yVarAst > 0:  # Do the astrometric variance correction
        S_R = np.fft.ifft2(R_hat * Kr_hat)
        gradRx, gradRy = np.gradient(S_R)
        fGradR = xVarAst * gradRx**2. + yVarAst * gradRy**2.
        S_N = np.fft.ifft2(N_hat * Kn_hat)
        gradNx, gradNy = np.gradient(S_N)
        fGradN = xVarAst * gradNx**2. + yVarAst * gradNy**2.

    S_var = np.sqrt(np.fft.ifftshift(np.fft.ifft2(var1c_hat + var2c_hat)) + fGradR + fGradN)
    S_var *= Fd

    S = np.fft.ifftshift(np.fft.ifft2(Kn_hat * N_hat - Kr_hat * R_hat))
    S *= Fd

    Pd = computeZogyDiffimPsf(im1, im2, im1_psf, im2_psf, sig1, sig2, Fr, Fn,
                              padSize=0)
    D_hat = (Fr * Pr_hat * N_hat - Fn * Pn_hat * R_hat)
    D_hat /= denom
    return S.real, S_var.real, Pd, Fd


def computeZogyScorrImageSpace(D, im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                               sig1=None, sig2=None, Fr=1., Fn=1., xVarAst=0., yVarAst=0.,
                               padSize=0):
    prereqs = computeZogyPrereqs(im1, im2, im1_psf, im2_psf,
                                 sig1, sig2, Fr, Fn, padSize=0)
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


def computeZogyScorr(im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                     sig1=None, sig2=None, Fr=1., Fn=1., xVarAst=0., yVarAst=0.,
                     D=None, padSize=0, inImageSpace=True):
    if inImageSpace:
        if D is None:
            D, _ = computeZogyImageSpace(im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                                         sig1=sig1, sig2=sig2, Fr=1., Fn=1., padSize=padSize)
        return computeZogyScorrImageSpace(D, im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                                          sig1=sig1, sig2=sig2, Fr=Fr, Fn=Fn,
                                          xVarAst=xVarAst, yVarAst=yVarAst, padSize=padSize)
    else:
        return computeZogyScorrFourierSpace(im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                                            sig1=sig1, sig2=sig2, Fr=Fr, Fn=Fn,
                                            xVarAst=xVarAst, yVarAst=yVarAst)


class Zogy(object):
    def __init__(self, im1, im2, im1_var, im2_var, im1_psf, im2_psf,
                 sig1=None, sig2=None, Fr=1., Fn=1., inImageSpace=False, padSize=7):
        self.im1, self.im2 = im1, im2
        self.im1_var, self.im2_var = im1_var, im2_var
        self.im1_psf, self.im2_psf = im1_psf, im2_psf
        self.sig1, self.sig2 = sig1, sig2
        self.Fr, self.Fn = Fr, Fn
        self.Fd = 1.
        self.inImageSpace = inImageSpace
        self.padSize = padSize
        self.D = self.D_var = self.Pd = self.S = self.S_var = None

        if self.sig1 is None:
            _, self.sig1, _, _ = computeClippedImageStats(im1)
        if self.sig2 is None:
            _, self.sig2, _, _ = computeClippedImageStats(im2)

    def _zogy(self):
        D, D_var = computeZogy(self.im1, self.im2, self.im1_var, self.im2_var,
                               im1_psf=self.im1_psf, im2_psf=self.im2_psf,
                               sig1=self.sig1, sig2=self.sig2,
                               Fr=self.Fr, Fn=self.Fn, inImageSpace=self.inImageSpace,
                               padSize=self.padSize)
        Pd = computeZogyDiffimPsf(self.im1, self.im2, self.im1_psf, self.im2_psf,
                                  self.sig1, self.sig2, self.Fr, self.Fn)
        return D, D_var, Pd

    def _zogyScorr(self, varAst=[0., 0.]):
        S, S_var, Pd, Fd = computeZogyScorr(self.im1, self.im2, self.im1_var, self.im2_var,
            im1_psf=self.im1_psf, im2_psf=self.im2_psf, sig1=self.sig1, sig2=self.sig2,
            Fr=self.Fr, Fn=self.Fn, xVarAst=varAst[0], yVarAst=varAst[1], # these are already variances.
            inImageSpace=self.inImageSpace, D=self.D, padSize=0)
        return S, S_var, Pd, Fd

    # TBD -- a lot of the D calc happens twice. Figure out a way to make it happen just once.
    def doZogy(self, computeScorr=False):
        self.D, self.D_var, self.Pd = self._zogy()

        if computeScorr:
            self.S, self.S_var, _, self.Fd = self._zogyScorr(varAst=[0., 0.])



class ZogyTask(Zogy):
    def __init__(self, im1, im2, inImageSpace=False):
        Zogy.__init__(self, im1.im, im2.im, im1.var, im2.var, im1.psf, im2.psf,
                      inImageSpace=inImageSpace)

    def doZogy(self, computeScorr=False):
        Zogy.doZogy(self, computeScorr=computeScorr)
        self.D = Exposure(self.D, self.Pd, self.D_var)
        if computeScorr:
            self.S = Exposure(self.S, self.Pd, self.S_var)
