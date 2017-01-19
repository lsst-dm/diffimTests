import numpy as np
import numpy.fft
import scipy.signal

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

# In all functions, im1 is R (reference, or template) and im2 is N (new, or science)
def ZOGYUtils(im1, im2, im1_psf, im2_psf, sig1=None, sig2=None, F_r=1., F_n=1., padSize=0):
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

    P_r = psf1 #im1_psf
    P_n = psf2 #im2_psf
    sigR = sig1
    sigN = sig2
    P_r_hat = numpy.fft.fft2(P_r)
    P_n_hat = numpy.fft.fft2(P_n)
    denom = np.sqrt((sigN**2 * F_r**2 * np.abs(P_r_hat)**2) + (sigR**2 * F_n**2 * np.abs(P_n_hat)**2))
    #denom = np.sqrt((sigN**2 * F_r**2 * P_r_hat**2) + (sigR**2 * F_n**2 * P_n_hat**2))

    return sigR, sigN, P_r_hat, P_n_hat, denom, P_r, P_n


# In all functions, im1 is R (reference, or template) and im2 is N (new, or science)
def performZOGY(im1, im2, im1_psf, im2_psf, sig1=None, sig2=None, F_r=1., F_n=1.):
    sigR, sigN, P_r_hat, P_n_hat, denom, _, _ = ZOGYUtils(im1, im2, im1_psf, im2_psf,
                                                          sig1, sig2, F_r, F_n, padSize=0)

    R_hat = numpy.fft.fft2(im1)
    N_hat = numpy.fft.fft2(im2)
    numerator = (F_r * P_r_hat * N_hat - F_n * P_n_hat * R_hat)
    d_hat = numerator / denom

    d = numpy.fft.ifft2(d_hat)
    D = numpy.fft.ifftshift(d.real)

    return D


# In all functions, im1 is R (reference, or template) and im2 is N (new, or science)
def performZOGYImageSpace(im1, im2, im1_psf, im2_psf, sig1=None, sig2=None, F_r=1., F_n=1., padSize=15):
    sigR, sigN, P_r_hat, P_n_hat, denom, padded_psf1, padded_psf2 = ZOGYUtils(im1, im2, im1_psf, im2_psf,
                                                                              sig1, sig2, F_r, F_n,
                                                                              padSize=padSize)
    delta = 0 #.1
    K_r_hat = (P_r_hat + delta) / (denom + delta)
    K_n_hat = (P_n_hat + delta) / (denom + delta)
    K_r = np.fft.ifft2(K_r_hat).real
    K_n = np.fft.ifft2(K_n_hat).real

    if padSize > 0:
        K_n = K_n[padSize:-padSize, padSize:-padSize]
        K_r = K_r[padSize:-padSize, padSize:-padSize]

    # Note these are reverse-labelled, this is CORRECT!
    im1c = scipy.ndimage.filters.convolve(im1, K_n, mode='constant', cval=np.nan)
    im2c = scipy.ndimage.filters.convolve(im2, K_r, mode='constant', cval=np.nan)
    D = im2c - im1c

    return D


## Also compute the diffim's PSF (eq. 14)
def computeZOGYDiffimPsf(im1, im2, im1_psf, im2_psf, sig1=None, sig2=None, F_r=1., F_n=1., padSize=0):
    sigR, sigN, P_r_hat, P_n_hat, denom, _, _ = ZOGYUtils(im1, im2, im1_psf, im2_psf,
                                                          sig1, sig2, F_r, F_n, padSize=padSize)

    F_D_numerator = F_r * F_n
    F_D_denom = np.sqrt(sigN**2 * F_r**2 + sigR**2 * F_n**2)
    F_D = F_D_numerator / F_D_denom

    P_d_hat_numerator = (F_r * F_n * P_r_hat * P_n_hat)
    P_d_hat = P_d_hat_numerator / (F_D * denom)

    P_d = np.fft.ifft2(P_d_hat)
    P_D = np.fft.ifftshift(P_d).real

    return P_D, F_D


# Compute the corrected ZOGY "S_corr" (eq. 25)
# Currently only implemented is V(S_N) and V(S_R)
# Want to implement astrometric variance Vast(S_N) and Vast(S_R)
def performZOGY_Scorr(im1, im2, var_im1, var_im2, im1_psf, im2_psf,
                      sig1=None, sig2=None, F_r=1., F_n=1., xVarAst=0., yVarAst=0., D=None, padSize=15):
    if D is None:
        D = performZOGYImageSpace(im1, im2, im1_psf, im2_psf, sig1, sig2, F_r, F_n, padSize=padSize)
    P_D, F_D = computeZOGYDiffimPsf(im1, im2, im1_psf, im2_psf, sig1, sig2, F_r, F_n)
    # P_r_hat = np.fft.fftshift(P_r_hat)  # Not sure why I need to do this but it seems that I do.
    # P_n_hat = np.fft.fftshift(P_n_hat)

    sigR, sigN, P_r_hat, P_n_hat, denom, _, _ = ZOGYUtils(im1, im2, im1_psf, im2_psf,
                                                          sig1, sig2, F_r, F_n, padSize=padSize)

    # Adjust the variance planes of the two images to contribute to the final detection
    # (eq's 26-29).
    k_r_hat = F_r * F_n**2 * np.conj(P_r_hat) * np.abs(P_n_hat)**2 / denom**2.
    k_n_hat = F_n * F_r**2 * np.conj(P_n_hat) * np.abs(P_r_hat)**2 / denom**2.

    k_r = np.fft.ifft2(k_r_hat)
    k_r = k_r.real  # np.abs(k_r).real #np.fft.ifftshift(k_r).real
    k_r = np.roll(np.roll(k_r, -1, 0), -1, 1)
    k_n = np.fft.ifft2(k_n_hat)
    k_n = k_n.real  # np.abs(k_n).real #np.fft.ifftshift(k_n).real
    k_n = np.roll(np.roll(k_n, -1, 0), -1, 1)
    if padSize > 0:
        k_n = k_n[padSize:-padSize, padSize:-padSize]
        k_r = k_r[padSize:-padSize, padSize:-padSize]
    var1c = scipy.ndimage.filters.convolve(var_im1, k_r**2., mode='constant', cval=np.nan)
    var2c = scipy.ndimage.filters.convolve(var_im2, k_n**2., mode='constant', cval=np.nan)

    fGradR = fGradN = 0.
    if xVarAst + yVarAst > 0:  # Do the astrometric variance correction
        S_R = scipy.ndimage.filters.convolve(im1, k_r, mode='constant', cval=np.nan)
        gradRx, gradRy = np.gradient(S_R)
        fGradR = xVarAst * gradRx**2. + yVarAst * gradRy**2.
        S_N = scipy.ndimage.filters.convolve(im2, k_n, mode='constant', cval=np.nan)
        gradNx, gradNy = np.gradient(S_N)
        fGradN = xVarAst * gradNx**2. + yVarAst * gradNy**2.

    PD_bar = np.fliplr(np.flipud(P_D))
    S = scipy.ndimage.filters.convolve(D, PD_bar, mode='constant', cval=np.nan) * F_D
    S_corr = S / np.sqrt(var1c + var2c + fGradR + fGradN)
    return S_corr, S, D, P_D, F_D, var1c, var2c
