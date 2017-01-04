import numpy as np
import scipy.stats


def gaussian2d(grid, m=None, s=None):
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
    # This is slow, so we use the func. below instead.
    if m is None:
        m = [0., 0.]
    if s is None:
        s = [1., 1.]
    cov = [[s[0], 0], [0, s[1]]]
    var = scipy.stats.multivariate_normal(mean=m, cov=cov)
    return var.pdf(grid)


def singleGaussian2d(x, y, xc, yc, sigma_x=1., sigma_y=1., theta=0., offset=0.):
    theta = (theta/180.) * np.pi
    cos_theta2, sin_theta2 = np.cos(theta)**2., np.sin(theta)**2.
    sigma_x2, sigma_y2 = sigma_x**2., sigma_y**2.
    a = cos_theta2/(2.*sigma_x2) + sin_theta2/(2.*sigma_y2)
    b = -(np.sin(2.*theta))/(4.*sigma_x2) + (np.sin(2.*theta))/(4.*sigma_y2)
    c = sin_theta2/(2.*sigma_x2) + cos_theta2/(2.*sigma_y2)
    xxc, yyc = x-xc, y-yc
    out = np.exp(-(a*(xxc**2.) + 2.*b*xxc*yyc + c*(yyc**2.)))
    if offset != 0.:
        out += offset
    out /= out.sum()
    return out


def doubleGaussian2d(x, y, xc, yc, sigma_x=1., sigma_y=1., theta=0., offset=0.,
                     a=0.9, sigma_x2=None, sigma_y2=None, theta2=None):
    sigma_x2 = sigma_x * 2. if sigma_x2 is None else sigma_x2
    sigma_y2 = sigma_y * 2. if sigma_y2 is None else sigma_y2
    theta2 = theta if theta2 is None else theta2

    g1 = a * singleGaussian2d(x, y, xc, yc, sigma_x1, sigma_y1, theta1, offset)
    g1 += (1-a) * singleGaussian2d(x, y, xc, yc, sigma_x2, sigma_y2, theta2, offset)
    g1 /= g1.sum()
    return g1


def moffat2d(x, y, xc, yc, fwhm=1., alpha=4.765):
    """Two dimensional Moffat.

    Parameters
    ----------
    amplitude : float
        Amplitude of the model.
    x_0, y_0 : float
        x, y position of the maximum of the Moffat model.
    gamma : float
        Core width of the Moffat model.
    alpha : float
        Power index of the Moffat model.
    """

    #fwhm = 2. * gamma * np.sqrt(2**(1./alpha) - 1)
    gamma = fwhm / (2. * np.sqrt(2**(1./alpha) - 1))
    rr_gg = ((x - xc)**2. + (y - yc)**2.) / gamma**2.
    out = (1 + rr_gg)**(-alpha)
    out /= out.sum()
    return out


def computeMoments(psf):
    xgrid, ygrid = np.meshgrid(np.arange(0, psf.shape[0]), np.arange(0, psf.shape[1]))
    xmoment = np.average(xgrid, weights=psf)
    ymoment = np.average(ygrid, weights=psf)
    return xmoment, ymoment

