import numpy as np


def padCenterOriginArray(A, newShape: tuple, onwardOp=True):
    """Zero pad an image where the origin is at the center and replace the
    origin to the corner as required by the periodic input of FFT. Implement also
    the inverse operation, crop the padding and re-center data.

    Parameters
    ----------
    A : `numpy.ndarray`
        An array to copy from.
    newShape : `tuple` of `int`
        The dimensions of the resulting array. For padding, the resulting array
        must be larger than A in each dimension. For the inverse operation this
        must be the original, before padding size of the array.
    onwardOp : bool, optional
        Selector of the padding (True) or its inverse (False) operation.

    Returns
    -------
    R : `numpy.ndarray`
        The padded or unpadded array with shape of `newShape` and the same dtype as A.

    Notes
    -----
    Supports n-dimension arrays. For odd dimensions, the splitting is rounded to
    put the center element into the new origin (eg. the center pixel of an odd sized
    kernel will be located at (0,0) ready for FFT).

    """
    R = np.zeros_like(A, shape=newShape)
    # The onward and inverse operations should round odd dimension halves at the opposite
    # sides to get the pixels back to their original positions.
    if onwardOp:
        firstHalves = [x//2 for x in A.shape]
        secondHalves = [x-y for x, y in zip(A.shape, firstHalves)]
    else:
        secondHalves = [x//2 for x in newShape]
        firstHalves = [x-y for x, y in zip(newShape, secondHalves)]

    # R[..., -firstHalf: , ... ] = A[..., :firstHalf, ...]
    firstAs = [slice(None, x) for x in firstHalves]
    firstRs = [slice(-x, None) for x in firstHalves]

    # R[..., :secondHalf , ... ] = A[..., -secondHalf:, ...]
    secondAs = [slice(-x, None) for x in secondHalves]
    secondRs = [slice(None, x) for x in secondHalves]

    nDim = len(A.shape)
    # Loop through all 2**nDim corners
    # (all combination of first and second halves regarding A)
    for c in range(1 << nDim):
        cornerA = []
        cornerR = []
        for i in range(nDim):
            if c & (1 << i):
                cornerA.append(firstAs[i])
                cornerR.append(firstRs[i])
            else:
                cornerA.append(secondAs[i])
                cornerR.append(secondRs[i])

        R[tuple(cornerR)] = A[tuple(cornerA)]
    return R


def padCornerArray(A, newShape: tuple, onwardOp=True):
    """Zero pad an array to the `right` . Implement also the inverse operation.
    """
    if onwardOp:
        # R[..., 0:A.shape[i], ...] = A
        cornerR = tuple([slice(None, x) for x in A.shape])
        cornerA = Ellipsis
    else:
        # R[...] = A[ ..., 0:newshape[i], ...]
        cornerR = Ellipsis
        cornerA = tuple([slice(None, x) for x in newShape])

    R = np.zeros_like(A, shape=newShape)
    R[cornerR] = A[cornerA]
    return R
