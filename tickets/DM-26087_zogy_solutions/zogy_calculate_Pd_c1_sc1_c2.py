import numpy as np


def calculatePd(psf1, varMean1, psf2, varMean2):
    """This is a code cutout from ZogyImagePsfMatchTask.calculateFourierDiffim().
    """
    F1 = 1.
    F2 = 1.
    var1F2Sq = varMean1*F2*F2
    var2F1Sq = varMean2*F1*F1
    # We need reals for comparison, also real operations are usually faster
    psfAbsSq1 = np.real(np.conj(psf1)*psf1)
    psfAbsSq2 = np.real(np.conj(psf2)*psf2)
    FdDenom = np.sqrt(var1F2Sq + var2F1Sq)  # one number

    # Secure positive limit to avoid floating point operations
    # resulting in exact zero
    tiny = np.finfo(psf1.dtype).tiny * 100
    sDenom = var1F2Sq*psfAbsSq2 + var2F1Sq*psfAbsSq1  # array, eq. (12)
    fltZero = sDenom < tiny
    nZero = np.sum(fltZero)
    print(f"Handling {nZero} both PSFs are zero points.")
    if nZero > 0:
        fltZero = np.nonzero(fltZero)  # We expect only a small fraction of such frequencies
        sDenom[fltZero] = tiny  # Avoid division problem but overwrite result anyway
    denom = np.sqrt(sDenom)  # array, eq. (13)

    Pd = FdDenom*psf1*psf2/denom  # Psf of D eq. (14)
    if nZero > 0:
        Pd[fltZero] = 0
    c1 = psf2/denom
    c2 = psf1/denom

    if nZero > 0:
        c1[fltZero] = 1./FdDenom
        c2[fltZero] = 1./FdDenom

    sc1 = np.conj(psf1)*psfAbsSq2/sDenom
    if nZero > 0:
        sc1[fltZero] = 0

    return Pd, c1, sc1, c2
