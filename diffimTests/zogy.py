import numpy as np
import scipy.signal

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
    denom = np.sqrt((sigN**2 * Fr**2 * np.abs(Pr_hat**2)) + (sigR**2 * Fn**2 * np.abs(Pn_hat**2)))

    return sigR, sigN, Pr_hat, Pn_hat, denom, Pr, Pn


# In all functions, im1 is R (reference, or template) and im2 is N (new, or science)
def performZogy(im1, im2, var_im1, var_im2, im1_psf, im2_psf, sig1=None, sig2=None, Fr=1., Fn=1.):
    sigR, sigN, Pr_hat, Pn_hat, denom, \
        _, _ = computeZogyPrereqs(im1, im2, im1_psf, im2_psf,
                                  sig1, sig2, Fr, Fn, padSize=0)
    Fd = Fr*Fn / np.sqrt(sigN**2 * Fr**2 + sigR**2 * Fn**2)

    # First do the image
    R_hat = np.fft.fft2(im1)
    N_hat = np.fft.fft2(im2)
    D_hat = (Fr * Pr_hat * N_hat - Fn * Pn_hat * R_hat)
    D_hat /= denom

    D = np.fft.ifft2(D_hat)
    D = np.fft.ifftshift(D.real) / Fd
    #D *= np.sqrt(sigR**2. + sigN**2.)  # Set to same scale as A&L

    # Do the exact same thing to the var images, except add them
    R_hat = np.fft.fft2(var_im1)
    N_hat = np.fft.fft2(var_im2)
    d_hat = (Fr * Pr_hat * N_hat + Fn * Pn_hat * R_hat)
    d_hat /= denom

    d = np.fft.ifft2(d_hat)
    D_var = np.fft.ifftshift(d.real)
    D_var *= np.sqrt(sigR**2. + sigN**2.)  # Set to same scale as A&L

    return D, D_var


# In all functions, im1 is R (reference, or template) and im2 is N (new, or science)
def performZogyImageSpace(im1, im2, var_im1, var_im2, im1_psf, im2_psf,
                          sig1=None, sig2=None, Fr=1., Fn=1., padSize=7):
    sigR, sigN, Pr_hat, Pn_hat, denom, \
        padded_psf1, padded_psf2 = computeZogyPrereqs(im1, im2, im1_psf, im2_psf,
                                                      sig1, sig2, Fr, Fn, padSize=padSize)
    Fd = Fr*Fn / np.sqrt(sigN**2 * Fr**2 + sigR**2 * Fn**2)

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
    D = (im2c - im1c)/ Fd
    #D *= np.sqrt(sigR**2. + sigN**2.)  # Set to same scale as A&L

    # Do the same convolutions to the variance images
    im1c = scipy.ndimage.filters.convolve(var_im1, Kn, mode='constant', cval=np.nan)
    im2c = scipy.ndimage.filters.convolve(var_im2, Kr, mode='constant', cval=np.nan)
    D_var = im2c + im1c
    D_var *= np.sqrt(sigR**2. + sigN**2.)  # Set to same scale as A&L

    return D, D_var


## Also compute the diffim's PSF (eq. 14)
def computeZogyDiffimPsf(im1, im2, im1_psf, im2_psf, sig1=None, sig2=None, Fr=1., Fn=1., padSize=0):
    sigR, sigN, Pr_hat, Pn_hat, denom, \
        _, _ = computeZogyPrereqs(im1, im2, im1_psf, im2_psf,
                                  sig1, sig2, Fr, Fn, padSize=padSize)

    FD_numerator = Fr * Fn
    FD_denom = np.sqrt(sigN**2 * Fr**2 + sigR**2 * Fn**2)
    FD = FD_numerator / FD_denom

    Pd_hat_numerator = (Fr * Fn * Pr_hat * Pn_hat)
    Pd_hat = Pd_hat_numerator / (FD * denom)

    Pd = np.fft.ifft2(Pd_hat)
    PD = np.fft.ifftshift(Pd).real

    return PD, FD


# Compute the corrected ZOGY "S_corr" (eq. 25)
# Currently only implemented is V(S_N) and V(S_R)
# Want to implement astrometric variance Vast(S_N) and Vast(S_R)
def performZogy_Scorr(im1, im2, var_im1, var_im2, im1_psf, im2_psf,
                      sig1=None, sig2=None, Fr=1., Fn=1., xVarAst=0., yVarAst=0., D=None,
                      inImageSpace=False, padSize=7):
    if D is None:
        if inImageSpace:
            D, _ = performZogyImageSpace(im1, im2, var_im1, var_im2, im1_psf, im2_psf,
                                         sig1=sig1, sig2=sig2, Fr=Fr, Fn=Fn, padSize=padSize)
        else:
            padSize = 0
            padSize0 = im1.shape[0]//2 - im1_psf.shape[0]//2
            padSize1 = im1.shape[1]//2 - im1_psf.shape[1]//2
            # Hastily assume the image is even-sized and the psf is odd...
            psf1 = np.pad(im1_psf, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
                          constant_values=0)
            psf2 = np.pad(im2_psf, ((padSize0, padSize0-1), (padSize1, padSize1-1)), mode='constant',
                          constant_values=0)
            D, _ = performZogy(im1, im2, var_im1, var_im2, psf1, psf2, sig1=sig1, sig2=sig2,
                               Fr=Fr, Fn=Fn)

    PD, FD = computeZogyDiffimPsf(im1, im2, im1_psf, im2_psf, sig1, sig2, Fr, Fn)

    sigR, sigN, Pr_hat, Pn_hat, denom, \
        _, _ = computeZogyPrereqs(im1, im2, im1_psf, im2_psf,
                                  sig1, sig2, Fr, Fn, padSize=padSize)

    # Adjust the variance planes of the two images to contribute to the final detection
    # (eq's 26-29).
    kr_hat = Fr * Fn**2. * np.conj(Pr_hat) * np.abs(Pn_hat)**2. / denom**2.
    kn_hat = Fn * Fr**2. * np.conj(Pn_hat) * np.abs(Pr_hat)**2. / denom**2.

    kr = np.fft.ifft2(kr_hat)
    kr = kr.real
    kr = np.roll(np.roll(kr, -1, 0), -1, 1)
    kn = np.fft.ifft2(kn_hat)
    kn = kn.real
    kn = np.roll(np.roll(kn, -1, 0), -1, 1)
    if padSize > 0:
        kn = kn[padSize:-padSize, padSize:-padSize]
        kr = kr[padSize:-padSize, padSize:-padSize]
    var1c = scipy.ndimage.filters.convolve(var_im1, kr**2., mode='constant', cval=np.nan)
    var2c = scipy.ndimage.filters.convolve(var_im2, kn**2., mode='constant', cval=np.nan)

    fGradR = fGradN = 0.
    if xVarAst + yVarAst > 0:  # Do the astrometric variance correction
        S_R = scipy.ndimage.filters.convolve(im1, kr, mode='constant', cval=np.nan)
        gradRx, gradRy = np.gradient(S_R)
        fGradR = xVarAst * gradRx**2. + yVarAst * gradRy**2.
        S_N = scipy.ndimage.filters.convolve(im2, kn, mode='constant', cval=np.nan)
        gradNx, gradNy = np.gradient(S_N)
        fGradN = xVarAst * gradNx**2. + yVarAst * gradNy**2.

    PD_bar = np.fliplr(np.flipud(PD))
    S = scipy.ndimage.filters.convolve(D, PD_bar, mode='constant', cval=np.nan) * FD
    S_var = np.sqrt(var1c + var2c + fGradR + fGradN)
    #S_corr = S #/ S_corr_var
    #return S_corr, S, S_corr_var, D, P_D, F_D, var1c, var2c
    S_var *= np.sqrt(sigR**2. + sigN**2.)  # Set to same scale as A&L (this was already done for S and D)
    return S, S_var, D, PD, FD, var1c, var2c
