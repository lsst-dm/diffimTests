import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.backends.backend_pdf import PdfPages
from lsst.ip.diffim.zogy import ZogyTask, ZogyConfig
import matplotlib.pyplot as plt
from array_freq_cut import *
from TwoD_gaussian_in_freq_space import *


def main():
    # Noise variance in the images
    varMean1 = 100
    varMean2 = 100
    # Photometric scaling
    F1 = 1.
    F2 = 1.

    # Here you can play with different input sigmas, in image space, in pixels
    # This notebook should not assume which one is the larger
    wSig1 = 4.
    wSig2 = 2.

    SigmaTuple = ((6.1, 6.0), (6.01, 6.0),
                  (4., 2.), (3., 2.), (4.3, 4.2), (3.3, 3.2),
                  (3., 3.1), (3.31, 3.3))

    with PdfPages("various_wSig_freq_masking.pdf") as pdfWriter:
        for (wSig1, wSig2) in SigmaTuple:
            print("wSig1={:.2f} wSig2={:.2f}".format(wSig1, wSig2))
            genFigures(wSig1, wSig2, F1, F2, varMean1, varMean2, pdfWriter)
            print("=========")


def genFigures(wSig1, wSig2, F1, F2, varMean1, varMean2, pdfWriter):
    config = ZogyConfig()
    config.scaleByCalibration = False
    task = ZogyTask(config=config)
    # The generated theoretical solutions. We use these values to replace the
    # DFT results above the cutting frequency.
    g_fc1, g_fc2 = calculateDirectModelFc1Fc2(1024, 1024, wSig1, wSig2, varMean1, varMean2)

    fig = plt.figure()
    fig.suptitle("$w\sigma_1=${:.2f}, $w\sigma_2=${:.2f}, var1={:.0f}, var2={:.0f}".
                 format(wSig1, wSig2, varMean1, varMean2))
    ax1 = fig.add_subplot(1, 1, 1)
    cs = ax1.imshow(g_fc1, origin='bottom', interpolation='none')
    ax1.set_title("Generated fc1".format(wSig1, wSig2))
    fig.colorbar(cs)
    pdfWriter.savefig(fig)
    plt.close(fig)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    cs = ax1.imshow(g_fc2, origin='bottom', interpolation='none')
    ax1.set_title("Generated fc2 low freq. at corner")
    fig.colorbar(cs)
    pdfWriter.savefig(fig)
    plt.close(fig)

    g_c1 = np.real(np.fft.ifft2(g_fc1))
    g_c2 = np.real(np.fft.ifft2(g_fc2))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # P2 = task.padCenterOriginArray(g_c1, (1024,1024), useInverse=True)
    # cs = ax.imshow(P2, interpolation='none', origin='bottom', cmap='RdBu_r',
    #                norm=matplotlib.colors.SymLogNorm(linthresh=1e-6,vmin=-0.01,vmax=0.01))
    # fig.colorbar(cs)
    # ax.set_title("Generated c1 (origin at center)")

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # P2 = task.padCenterOriginArray(g_c2, (1024, 1024), useInverse=True)
    # cs = ax.imshow(P2, interpolation='none', origin='bottom', cmap='RdBu_r',
    #                norm=matplotlib.colors.SymLogNorm(linthresh=1e-6,vmin=-0.01,vmax=0.01))
    # fig.colorbar(cs)
    # ax.set_title("Generated c2 (origin at center)")

    ### Now calculate where to cut the DFT solutions.

    # We can cut at the frequencies where the solutions reach their constant limit values
    fr1, fr2 = calculateCutFrequencies(wSig1, wSig2, varMean1, varMean2, limit=0.999999)
    print("Theoretical solution in Fourier space converged at:", fr1, fr2)
    print("in pixels: {:.0f} {:.0f}".format(fr1 * 1024, fr2 * 1024))
    # But we should cut at a lower frequency if the input PSF tail is reached at a lower frequency
    frp1 = calculateGaussianCutFrequency(wSig1, limit=0.99999)
    frp2 = calculateGaussianCutFrequency(wSig2, limit=0.99999)
    print("PSF tails reached in Fourier space at:", frp1, frp2)
    print("in pixels: {:.0f} {:.0f}".format(frp1 * 1024, frp2 * 1024))

    ## 31x31 PSFs padded and FFTd
    # Pad a usual 31x31 pix. PSF then FFT them

    A = calculate2dGaussianArray(31, 31, wSig1)
    A /= np.sum(A)
    pA = task.padCenterOriginArray(A,(1024,1024))
    psf1 = np.fft.fft2(pA)
    B = calculate2dGaussianArray(31, 31, wSig2)
    B /= np.sum(B)
    pB = task.padCenterOriginArray(B,(1024,1024))
    psf2 = np.fft.fft2(pB)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # cs = ax1.imshow(pA, origin='bottom', interpolation='none',  cmap='Blues',
    #                 norm=matplotlib.colors.LogNorm(vmin=1e-10,vmax=0.1))
    # ax2 = fig.add_subplot(1, 2, 2)
    # cs = ax2.imshow(pB, origin='bottom', interpolation='none',  cmap='Blues',
    #                 norm=matplotlib.colors.LogNorm(vmin=1e-10,vmax=0.1))
    # fig.colorbar(cs, ax=[ax1, ax2])
    # ax1.set_xlim(0, 50)
    # ax1.set_ylim(0, 50)
    # ax2.set_xlim(0, 50)
    # ax2.set_ylim(0, 50)
    # fig.suptitle("Padded shifted PSF (corner) before FFT")

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    cs = ax1.imshow(psf1.real, origin='bottom', interpolation='none',  cmap='RdBu_r',
                    norm=matplotlib.colors.SymLogNorm(linthresh=1e-8,vmin=-1,vmax=1))
    ax1.set_title("psf1")
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.imshow(psf2.real, origin='bottom', interpolation='none',  cmap='RdBu_r',
                    norm=matplotlib.colors.SymLogNorm(linthresh=1e-8,vmin=-1,vmax=1))
    ax2.set_title("psf2")
    fig.colorbar(cs, ax=[ax1, ax2])
    fig.suptitle("FFTd PSFs on log scale")
    pdfWriter.savefig(fig)
    plt.close(fig)

    # There are some small negative values around in the frequency space PSFs.

    var1F2Sq = varMean1*F2*F2
    var2F1Sq = varMean2*F1*F1
    FdDenom = np.sqrt(var1F2Sq + var2F1Sq)  # one number
    # We need reals for comparison, also real operations are usually faster
    # Psf absolute squared
    psfAbsSq1 = np.real(np.conj(psf1)*psf1)
    psfAbsSq2 = np.real(np.conj(psf2)*psf2)
    sDenom = var1F2Sq*psfAbsSq2 + var2F1Sq*psfAbsSq1  # array, eq. (12)
    # sDenom close to zero check here in the code, here we ignore, we won't hit division by zero
    denom = np.sqrt(sDenom)  # array, eq. (13)

    # sDenom: The squared denominator in the difference image calculation

    fPd = FdDenom*psf1*psf2/denom  # Psf of D eq. (14)
    fc1 = psf2/denom
    fc2 = psf1/denom

    # print(FdDenom)

    # Check all are real.
    assert np.all(fc2.imag == 0)
    assert np.all(fc1.imag == 0)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # cs = ax1.imshow(fc1.real, origin='bottom', interpolation='none', cmap='RdBu_r',
    #                 norm=matplotlib.colors.SymLogNorm(linthresh=1e-10,vmin=-0.1,vmax=0.1))
    # ax1.set_title("fc1 (origin at corner)")
    # ax2 = fig.add_subplot(1, 2, 2)
    # cs = ax2.imshow(fc2.real, origin='bottom', interpolation='none', cmap='RdBu_r',
    #                 norm=matplotlib.colors.SymLogNorm(linthresh=1e-10,vmin=-0.1,vmax=0.1))
    # ax2.set_title("fc2 (origin at corner)")
    # fig.colorbar(cs, ax=[ax1, ax2])
    # print(f"fc1 min {np.min(fc1.real)}")
    # print(f"fc1 max {np.max(fc1.real)}")
    # print(f"fc2 min {np.min(fc2.real)}")
    # print(f"fc2 max {np.max(fc1.real)}")



    # The matching kernel for the wider input PSF

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    cs = ax1.imshow(fc1.real, origin='bottom', interpolation='none')
    ax1.set_title("fc1 low freq. at corner")
    fig.colorbar(cs)
    pdfWriter.savefig(fig)
    plt.close(fig)

    if fr1 < frp2:
        print("fc1 is cut at the limit value of the theoretical fc1")
    else:
        print("fc1 is cut at the tail of PSF2")

    rN = np.minimum(fr1, frp2) * 1024
    freqFlt1 = makeEllipseQuartersMaskArray((1024, 1024), rN , rN)

    fig = plt.figure()
    if fr1 < frp2:
        fig.suptitle("fc1 is cut at the limit value of the theoretical fc1")
    else:
        fig.suptitle("fc1 is cut at the tail of PSF2")
    ax1 = fig.add_subplot(1, 1, 1)
    cs = ax1.imshow(np.asarray(freqFlt1, dtype=int), origin='bottom', interpolation='none')
    ax1.set_title("fc1 filter $r_{{1/px}}=${:.0f}".format(rN))
    pdfWriter.savefig(fig)
    plt.close(fig)

    # ========
    c1 = np.real(np.fft.ifft2(fc1))
    c2 = np.real(np.fft.ifft2(fc2))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    P2 = task.padCenterOriginArray(c1, (1024,1024), useInverse=True)
    cs = ax.imshow(P2, interpolation='none', origin='bottom', cmap='RdBu_r',
                   norm=matplotlib.colors.SymLogNorm(linthresh=1e-6,vmin=-0.01,vmax=0.01))
    fig.colorbar(cs)
    ax.set_title("Uncleaned c1 (origin at center)")
    pdfWriter.savefig(fig)
    plt.close(fig)
    # And for the narrower one

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    P2 = task.padCenterOriginArray(c2, (1024,1024), useInverse=True)
    cs = ax.imshow(P2, interpolation='none', origin='bottom', cmap='RdBu_r',
                   norm=matplotlib.colors.SymLogNorm(linthresh=1e-6,vmin=-0.01,vmax=0.01))
    fig.colorbar(cs)
    ax.set_title("Uncleaned c2 (origin at center)")
    pdfWriter.savefig(fig)
    plt.close(fig)

    # =====

    freqFlt1 = np.logical_not(freqFlt1)
    fc1[freqFlt1] = g_fc1[freqFlt1]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    cs = ax1.imshow(fc1.real, origin='bottom', interpolation='none')
    ax1.set_title("Replaced fc1")
    fig.colorbar(cs)
    pdfWriter.savefig(fig)
    plt.close(fig)

    # The matching kernel for the narrower input PSF

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    cs = ax1.imshow(fc2.real, origin='bottom', interpolation='none')
    ax1.set_title("fc2 low freq. at corner")
    fig.colorbar(cs)
    pdfWriter.savefig(fig)
    plt.close(fig)

    if fr2 < frp1:
        print("fc2 is cut at the limit value of the theoretical fc2")
    else:
        print("fc2 is cut at the tail of PSF1")

    rN = np.minimum(fr2, frp1) * 1024
    freqFlt2 = makeEllipseQuartersMaskArray((1024, 1024), rN , rN)

    fig = plt.figure()
    if fr2 < frp1:
        fig.suptitle("fc2 is cut at the limit value of the theoretical fc2")
    else:
        fig.suptitle("fc2 is cut at the tail of PSF1")
    ax1 = fig.add_subplot(1, 1, 1)
    cs = ax1.imshow(np.asarray(freqFlt2, dtype=int), origin='bottom', interpolation='none')
    ax1.set_title("fc2 filter $r_{{1/px}}=${:.0f}".format(rN))
    pdfWriter.savefig(fig)
    plt.close(fig)


    freqFlt2 = np.logical_not(freqFlt2)
    fc2[freqFlt2] = g_fc2[freqFlt2]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    cs = ax1.imshow(fc2.real, origin='bottom', interpolation='none')
    ax1.set_title("Replaced fc2")
    fig.colorbar(cs)
    pdfWriter.savefig(fig)
    plt.close(fig)


    Pd = np.real(np.fft.ifft2(fPd))
    c1 = np.real(np.fft.ifft2(fc1))
    c2 = np.real(np.fft.ifft2(fc2))

    # The matching kernel for the wider input back in image space

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    P2 = task.padCenterOriginArray(c1, (1024,1024), useInverse=True)
    cs = ax.imshow(P2, interpolation='none', origin='bottom', cmap='RdBu_r',
                   norm=matplotlib.colors.SymLogNorm(linthresh=1e-6,vmin=-0.01,vmax=0.01))
    fig.colorbar(cs)
    ax.set_title("Cleaned c1 (origin at center)")
    pdfWriter.savefig(fig)
    plt.close(fig)
    # And for the narrower one

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    P2 = task.padCenterOriginArray(c2, (1024,1024), useInverse=True)
    cs = ax.imshow(P2, interpolation='none', origin='bottom', cmap='RdBu_r',
                   norm=matplotlib.colors.SymLogNorm(linthresh=1e-6,vmin=-0.01,vmax=0.01))
    fig.colorbar(cs)
    ax.set_title("Cleaned c2 (origin at center)")
    pdfWriter.savefig(fig)
    plt.close(fig)
    # PSF for the difference image

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    P2 = task.padCenterOriginArray(Pd, (1024,1024), useInverse=True)
    cs = ax.imshow(P2, interpolation='none', origin='bottom', cmap='RdBu_r',
                   norm=matplotlib.colors.SymLogNorm(linthresh=1e-6,vmin=-0.05,vmax=0.05))
    fig.colorbar(cs)
    ax.set_title("Pd (origin at center) same scale as c1, c2")
    pdfWriter.savefig(fig)
    plt.close(fig)

if __name__ == "__main__":
    main()
