import numpy as np
import scipy.stats
#import pandas as pd  # We're going to store the results as pandas dataframes.


class sizeme():
    """ Class to change html fontsize of object's representation"""
    def __init__(self, ob, size=50, height=120):
        self.ob = ob
        self.size = size
        self.height = height
    def _repr_html_(self):
        repl_tuple = (self.size, self.height, self.ob._repr_html_())
        return u'<span style="font-size:{0}%; line-height:{1}%">{2}</span>'.format(*repl_tuple)


def getImageGrid(im):
    xim = np.arange(np.int(-np.floor(im.shape[0]/2.)), np.int(np.floor(im.shape[0]/2)))
    yim = np.arange(np.int(-np.floor(im.shape[1]/2.)), np.int(np.floor(im.shape[1]/2)))
    x0im, y0im = np.meshgrid(xim, yim)
    return x0im, y0im


def mad(data, axis=None):
    """ Median absolute deviation"""
    return np.median(np.absolute(data - np.median(data, axis)), axis)


def computeClippedImageStats(im, low=3, high=3, ignore=None):
    im = im[~(np.isnan(im) | np.isinf(im))]
    if ignore is not None:
        for i in ignore:
            im = im[im != i]
    tmp = im
    if low != 0 and high != 0 and tmp.min() != tmp.max():
        _, low, upp = scipy.stats.sigmaclip(tmp, low=low, high=high)
        if not np.isnan(low) and not np.isnan(upp) and low != upp:
            tmp = im[(im > low) & (im < upp)]
    mean1 = np.nanmean(tmp)
    sig1 = np.nanstd(tmp)
    return mean1, sig1, np.nanmin(im), np.nanmax(im)


def computePixelCovariance(diffim, diffim2=None):
    diffim = diffim/diffim.std()
    shifted_imgs2 = None
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
    #out = np.corrcoef(shifted_imgs)
    if diffim2 is not None:
        shifted_imgs2 = [
            diffim2,
            np.roll(diffim2, 1, 0), np.roll(diffim2, -1, 0), np.roll(diffim2, 1, 1), np.roll(diffim2, -1, 1),
            np.roll(np.roll(diffim2, 1, 0), 1, 1), np.roll(np.roll(diffim2, 1, 0), -1, 1),
            np.roll(np.roll(diffim2, -1, 0), 1, 1), np.roll(np.roll(diffim2, -1, 0), -1, 1),
            np.roll(diffim2, 2, 0), np.roll(diffim2, -2, 0), np.roll(diffim2, 2, 1), np.roll(diffim2, -2, 1),
            np.roll(diffim2, 3, 0), np.roll(diffim2, -3, 0), np.roll(diffim2, 3, 1), np.roll(diffim2, -3, 1),
            np.roll(diffim2, 4, 0), np.roll(diffim2, -4, 0), np.roll(diffim2, 4, 1), np.roll(diffim2, -4, 1),
            np.roll(diffim2, 5, 0), np.roll(diffim2, -5, 0), np.roll(diffim2, 5, 1), np.roll(diffim2, -5, 1),
        ]
        shifted_imgs2 = np.vstack([i.flatten() for i in shifted_imgs2])
    out = np.cov(shifted_imgs, shifted_imgs2, bias=1)
    tmp2 = out.copy()
    np.fill_diagonal(tmp2, np.NaN)
    stat = np.nansum(tmp2)/np.sum(np.diag(out))  # print sum of off-diag / sum of diag
    return out, stat


import functools

def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer
