import numpy as np
import numpy.fft


# Updated ALZC kernel and PSF computation (from ip_diffim.imageDecorrelation)
# This should eventually replace computeCorrectionKernelALZC and
# computeCorrectedDiffimPsfALZC

def computeDecorrelationKernel(kappa, tvar=0.04, svar=0.04, preConvKernel=None, delta=0.):
    """! Compute the Lupton/ZOGY post-conv. kernel for decorrelating an
    image difference, based on the PSF-matching kernel.
    @param kappa  A matching kernel 2-d numpy.array derived from Alard & Lupton PSF matching
    @param tvar   Average variance of template image used for PSF matching
    @param svar   Average variance of science image used for PSF matching
    @param preConvKernel   A pre-convolution kernel applied to im1 prior to A&L PSF matching
    @return a 2-d numpy.array containing the correction kernel

    @note As currently implemented, kappa is a static (single, non-spatially-varying) kernel.
    """
    #mk_center = np.unravel_index(np.argmax(kappa), kappa.shape)
    kappa = fixOddKernel(kappa)
    kft = numpy.fft.fft2(kappa)
    pc = pcft = 1.0
    if preConvKernel is not None:
        pc = fixOddKernel(preConvKernel)
        pcft = numpy.fft.fft2(pc)

    kft = np.sqrt((svar + tvar + delta) / (svar * np.abs(pcft)**2. + tvar * np.abs(kft)**2. + delta))
    #if preConvKernel is not None:
    #    kft = numpy.fft.fftshift(kft)  # I can't figure out why we need to fftshift sometimes but not others.
    pck = numpy.fft.ifft2(kft)
    #if np.argmax(pck.real) == 0:  # I can't figure out why we need to ifftshift sometimes but not others.
    #    pck = numpy.fft.ifftshift(pck.real)
    fkernel = fixEvenKernel(pck.real)

    # Make the center of the fkernel be the same as the "center" of the matching kernel.
    #pck_center = np.unravel_index(np.argmax(fkernel), fkernel.shape)
    #tmp = np.roll(fkernel, mk_center[0]-pck_center[0], axis=0)
    #tmp = np.roll(tmp, mk_center[1]-pck_center[1], axis=1)
    #fkernel = tmp

    # I think we may need to "reverse" the PSF, as in the ZOGY (and Kaiser) papers...
    # This is the same as taking the complex conjugate in Fourier space before FFT-ing back to real space.
    if False:  # TBD: figure this out. For now, we are turning it off.
        fkernel = fkernel[::-1, ::-1]

    return fkernel


def computeCorrectedDiffimPsf(kappa, psf, svar=0.04, tvar=0.04):
    """! Compute the (decorrelated) difference image's new PSF.
    new_psf = psf(k) * sqrt((svar + tvar) / (svar + tvar * kappa_ft(k)**2))

    @param kappa  A matching kernel array derived from Alard & Lupton PSF matching
    @param psf    The uncorrected psf array of the science image (and also of the diffim)
    @param svar   Average variance of science image used for PSF matching
    @param tvar   Average variance of template image used for PSF matching
    @return a 2-d numpy.array containing the new PSF
    """
    def post_conv_psf_ft2(psf, kernel, svar, tvar):
        # Pad psf or kernel symmetrically to make them the same size!
        # Note this assumes they are both square (width == height)
        if psf.shape[0] < kernel.shape[0]:
            diff = (kernel.shape[0] - psf.shape[0]) // 2
            psf = np.pad(psf, (diff, diff), mode='constant')
        elif psf.shape[0] > kernel.shape[0]:
            diff = (psf.shape[0] - kernel.shape[0]) // 2
            kernel = np.pad(kernel, (diff, diff), mode='constant')

        psf = fixOddKernel(psf)
        psf_ft = numpy.fft.fft2(psf)
        kernel = fixOddKernel(kernel)
        kft = numpy.fft.fft2(kernel)
        out = psf_ft * np.sqrt((svar + tvar) / (svar + tvar * np.abs(kft)**2.))
        return out

    def post_conv_psf(psf, kernel, svar, tvar):
        kft = post_conv_psf_ft2(psf, kernel, svar, tvar)
        out = numpy.fft.ifft2(kft)
        return out

    pcf = post_conv_psf(psf=psf, kernel=kappa, svar=svar, tvar=tvar)
    pcf = fixEvenKernel(pcf)
    pcf = pcf.real / pcf.real.sum()
    return pcf


def fixOddKernel(kernel):
    """! Take a kernel with odd dimensions and make them even for FFT

    @param kernel a numpy.array
    @return a fixed kernel numpy.array. Returns a copy if the dimensions needed to change;
    otherwise just return the input kernel.
    """
    # Note this works best for the FFT if we left-pad
    out = kernel
    changed = False
    if (out.shape[0] % 2) == 1:
        out = np.pad(out, ((1, 0), (0, 0)), mode='constant')
        changed = True
    if (out.shape[1] % 2) == 1:
        out = np.pad(out, ((0, 0), (1, 0)), mode='constant')
        changed = True
    if changed:
        out *= (np.mean(kernel) / np.mean(out))  # need to re-scale to same mean for FFT
    return out


def fixEvenKernel(kernel):
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
