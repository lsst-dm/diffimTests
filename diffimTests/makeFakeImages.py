import numpy as np

from .psf import makePsf


__all__ = ['makeFakeImages']

# Make the two "images". im1 is the template, im2 is the science
# image.
# NOTE: having sources near the edges really messes up the
# fitting (probably because of the convolution). So make sure no
# sources are near the edge.
# NOTE: also it seems that having the variable source with a large
# flux increase also messes up the fitting (seems to lead to
# overfitting -- perhaps to the source itself). This might be fixed by
# adding more constant sources.
# varFlux1 is the flux of variable sources in im1 (template). If zero, then the variable sources are
#  "new" sources in im2.
# varFlux2 is the flux of variable sources in im2 (science img)

# Note that n_sources has to be >= len(varFlux2). In fact, to get a desired number of static
# sources, you want to have n_sources = (# desired static sources) + len(varFlux2)

# TBD: (1) make no-noise template (DONE - templateNoNoise)
#      (2) allow enforce sky-limited (i.e., no shot noise in variance from stars) (DONE - skyLimited)
#      (3) add variable relative background by polynomial;
#      (4) add randomness to PSF shapes of stars
#      (5) add doubleGaussian2d as possible PSF shape (DONE - psfType='doubleGaussian' (can also use 'moffat'))


def makeFakeImages(imSize=(512, 512), sky=[300., 300.], psf1=[1.6, 1.6], psf2=[1.8, 2.2],
                   theta1=0., theta2=-45., psfType='gaussian', offset=[0., 0.], randAstromVariance=0.,
                   psf_yvary_factor=0., varFlux1=0, varFlux2=np.repeat(750, 50), im2background=0.,
                   n_sources=1500, templateNoNoise=False, skyLimited=False, sourceFluxRange=(250, 60000),
                   variablesNearCenter=False, avoidBorder=1.1, avoidAllOverlaps=0.,
                   sourceFluxDistrib='exponential', psfSize=21, seed=66, fast=True, verbose=False):
    if seed is not None:  # use None if you set the seed outside of this func.
        np.random.seed(seed)

    if not hasattr(sky, "__len__"):
        sky = [sky, sky]
    if not hasattr(psf1, "__len__"):
        psf1 = [psf1, psf1]
    if not hasattr(psf2, "__len__"):
        psf2 = [psf2, psf2]
    if not hasattr(offset, "__len__"):
        offset = [offset, offset]

    # Input PSF is in sigmas, in units of pixels.
    # For LSST, avg. seeing is ~0.7" or ~3.5 pixels (FWHM), or for sigma is 3.5/2.35482 = 1.48.
    # Thus the default PSF is slightly worse than average (template) and fairly bad (science)
    if verbose:
        print 'Template PSF:', psf1, theta1
        print 'Science PSF:', psf2, theta2
        print np.sqrt(psf2[0]**2 - psf1[1]**2)
        print 'Offset:', offset

    xim = np.arange(-imSize[0]//2, imSize[0]//2, 1)
    yim = np.arange(-imSize[1]//2, imSize[1]//2, 1)
    x0im, y0im = np.meshgrid(xim, yim)

    # Compute the star fluxes
    if sourceFluxDistrib == 'uniform':
        fluxes = np.random.uniform(sourceFluxRange[0], sourceFluxRange[1], n_sources)
    elif sourceFluxDistrib == 'exponential':
        # More realistic (euclidean), # of stars goes as 10**(0.6mag) so decreases by about 3.98x per increasing 1 magnitude
        # Looking toward the disk/bulge, this probably decreases to ~3.
        # This means # of stars increases about ~3x per decreasing ~2.512x in flux.
        # So we use: n = flux**(-3./2.512)
        fluxes = np.exp(np.linspace(np.log(sourceFluxRange[0]), np.log(sourceFluxRange[1])))
        n_flux = (np.array(fluxes)/sourceFluxRange[1])**(-3./2.512)
        samples = np.array([])
        tries = 0
        while len(samples) < n_sources and tries < 100:
            i_choice = np.random.randint(0, len(fluxes), 1000)
            f_choice = fluxes[i_choice]
            n_choice = n_flux[i_choice]
            rand = np.random.rand(len(i_choice)) * np.max(n_choice)
            chosen = rand <= n_choice
            f_chosen = f_choice[chosen]
            samples = np.append(samples, f_chosen)
            tries += 1

        fluxes = samples[0:n_sources]

    #fluxes = np.sort(fluxes)[::-1]

    # Place the stars
    border = 2  #5
    if avoidBorder * psfSize > border: # number of pixels to avoid putting sources near image boundary
        border = int(avoidBorder * psfSize)
    if avoidAllOverlaps == 0.:  # Don't care where stars go, just add them randomly.
        xposns = np.random.uniform(xim.min()+border, xim.max()-border, n_sources)
        yposns = np.random.uniform(yim.min()+border, yim.max()-border, n_sources)
    else:  # `avoidAllOverlaps` gives radius (pixels) of exclusion
        xposns = np.random.uniform(xim.min()+border, xim.max()-border, 1)
        yposns = np.random.uniform(yim.min()+border, yim.max()-border, 1)
        nTriedAndFailed = 0
        maxTries = 100
        for i in range(n_sources-1):
            xpos, ypos = xposns[-1], yposns[-1]
            dists = np.sqrt((xpos - xposns)**2. + (ypos - yposns)**2.)
            notTooManyTries = 0
            while((dists.min() < avoidAllOverlaps) and (notTooManyTries < maxTries)):
                xpos = np.random.uniform(xim.min()+border, xim.max()-border, 1)[0]
                ypos = np.random.uniform(yim.min()+border, yim.max()-border, 1)[0]
                dists = np.sqrt((xpos - xposns)**2. + (ypos - yposns)**2.)
                notTooManyTries += 1
            xposns = np.append(xposns, [xpos])
            yposns = np.append(yposns, [ypos])
            if notTooManyTries > 99:
                nTriedAndFailed += 1
            if nTriedAndFailed > 20:  # for very crowded fields, don't try too hard.
                maxTries = 1
        xposns = np.array(xposns)
        yposns = np.array(yposns)

    fluxSortedInds = np.argsort(xposns**2. + yposns**2.)[::-1]

    if not hasattr(varFlux1, "__len__"):
        varFlux1 = [varFlux1]
    if not hasattr(varFlux2, "__len__"):
        varFlux2 = [varFlux2]
    if len(varFlux1) == 1:
        varFlux1 = np.repeat(varFlux1[0], len(varFlux2))
    if variablesNearCenter:
        # Make the sources closest to the center of the image the ones that increases in flux
        inds = fluxSortedInds[:len(varFlux2)]
    else:  # Just choose random ones
        inds = np.arange(len(varFlux2))
        np.random.shuffle(inds)
    #print inds, xposns[inds], yposns[inds]

    ## Need to add poisson noise of stars as well...
    im1 = np.random.poisson(sky[0], size=x0im.shape).astype(float)  # sigma of template
    if templateNoNoise:
        im1[:] = sky[0]
    im2 = np.random.poisson(sky[1], size=x0im.shape).astype(float)  # sigma of science image

    var_im1 = im1.copy()
    if templateNoNoise:
        #var_im1[:] = 1.  # setting it to a single value just leads to all kinds of badness
        var_im1[:] = np.random.poisson(2., size=x0im.shape).astype(float)
    var_im2 = im2.copy()

    # variation in y-width of psf in science image across (x-dim of) image
    # A good value if you turn it on is 0.2.
    psf2_yvary = psf_yvary_factor * (yim.mean() - yposns) / yim.max()
    if verbose:
        print 'PSF y spatial-variation:', psf2_yvary.min(), psf2_yvary.max()
    # psf2_yvary[:] = 1.1  # turn it off for now, just add a constant 1.1 pixel horizontal width

    astromNoiseX = astromNoiseY = np.zeros(len(fluxes))
    if randAstromVariance > 0.:
        astromNoiseX = np.random.normal(0., randAstromVariance, len(fluxes))
        astromNoiseY = np.random.normal(0., randAstromVariance, len(fluxes))

    starSize = psfSize + 1  # make stars using "psf's" of this size (instead of whole image)
    xstar = np.arange(-starSize+1, starSize, 1)
    ystar = xstar.copy()
    y0star, x0star = np.meshgrid(xstar, ystar)

    # Allow for 2 psfType's - for template and science img. separately
    if isinstance(psfType, str) or not hasattr(psfType, "__len__"):
        psfType = [psfType, psfType]

    nDidFast = 0
    varSourceInd = 0
    fluxes2 = fluxes.copy()
    for i in fluxSortedInds:
        if i in inds:
            flux = varFlux1[varSourceInd]
        else:
            flux = fluxes[i]
        fluxes[i] = flux

        if fast and xposns[i] > -imSize[0]//2+starSize and yposns[i] > -imSize[1]//2+starSize and \
           xposns[i] < imSize[0]//2 - starSize and yposns[i] < imSize[1]//2 - starSize:
            nDidFast += 1
            offset1 = [yposns[i]-np.floor(yposns[i]),
                       xposns[i]-np.floor(xposns[i])]
            tmp = flux * makePsf(psfType=psfType[0], sigma=psf1, theta=theta1, offset=offset1,
                                 x0=x0star, y0=y0star, psfSize=starSize)
            offset2 = [xposns[i]+imSize[0]//2, yposns[i]+imSize[1]//2]
            tmp[tmp < 0.] = 0.  # poisson cant handle <0.
            if not templateNoNoise:
                tmp = np.random.poisson(tmp, size=tmp.shape).astype(float)
            im1[(offset2[1]-starSize+1):(offset2[1]+starSize),
                (offset2[0]-starSize+1):(offset2[0]+starSize)] += tmp
            if not skyLimited:
                var_im1[(offset2[1]-starSize+1):(offset2[1]+starSize),
                        (offset2[0]-starSize+1):(offset2[0]+starSize)] += tmp
        else:
            tmp = flux * makePsf(psfType=psfType[0], sigma=psf1, theta=theta1,
                                 offset=[xposns[i], yposns[i]], x0=x0im, y0=y0im, psfSize=0)
            tmp[tmp < 0.] = 0.  # poisson cant handle <0.
            if not templateNoNoise:
                tmp = np.random.poisson(tmp, size=tmp.shape).astype(float)
            im1 += tmp
            if not skyLimited:
                var_im1 += tmp

        if i in inds:
            vf2 = varFlux2[varSourceInd]
            if vf2 < 1:  # option to input it as fractional flux change
                vf2 = flux * vf2
            changedCentroid = (xposns[i]+imSize[0]//2, yposns[i]+imSize[1]//2)
            if verbose:
                print 'Variable source:', i, changedCentroid[0], changedCentroid[1], flux, flux + vf2
            flux += vf2
            fluxes2[i] = flux
            varSourceInd += 1
        xposn = xposns[i] + offset[0] + astromNoiseX[i]
        yposn = yposns[i] + offset[1] + astromNoiseY[i]
        psftmp = [psf2[0], psf2[1] + psf2_yvary[i]]

        if fast and xposn > -imSize[0]//2+starSize and yposn > -imSize[1]//2+starSize and \
           xposn < imSize[0]//2 - starSize and yposn < imSize[1]//2 - starSize:
            nDidFast += 1
            offset1 = [yposn-np.floor(yposn), xposn-np.floor(xposn)]
            tmp = flux * makePsf(psfType=psfType[1], sigma=psftmp, theta=theta2,
                                 offset=offset1, x0=x0star, y0=y0star, psfSize=starSize)
            offset2 = [xposn+imSize[0]//2, yposn+imSize[1]//2]
            tmp[tmp < 0.] = 0.  # poisson cant handle <0.
            tmp = np.random.poisson(tmp, size=tmp.shape).astype(float)
            im2[(offset2[1]-starSize+1):(offset2[1]+starSize),
                (offset2[0]-starSize+1):(offset2[0]+starSize)] += tmp
            if not skyLimited:
                var_im2[(offset2[1]-starSize+1):(offset2[1]+starSize),
                        (offset2[0]-starSize+1):(offset2[0]+starSize)] += tmp
        else:
            tmp = flux * makePsf(psfType=psfType[1], sigma=psftmp, theta=theta2,
                                 offset=[xposn, yposn], x0=x0im, y0=y0im, psfSize=0)
            tmp[tmp < 0.] = 0.  # poisson cant handle <0.
            tmp = np.random.poisson(tmp, size=tmp.shape).astype(float)
            im2 += tmp
            if not skyLimited:
                var_im2 += tmp

    im1 -= sky[0]
    im2 -= sky[1]

    # Add a (constant, for now) background offset to im2
    if im2background != 0.:  # im2background = 10.
        print 'Background:', im2background
        im2 += im2background

    if psfSize is None:
        psfSize = imSize

    im1_psf = makePsf(psfType=psfType[0], sigma=psf1, theta=theta1, psfSize=psfSize)
    #im2_psf = makePsf(psfSize, psf2, theta2, offset)
    # Don't include any astrometric "error" in the PSF, see how well the diffim algo. handles it.
    im2_psf = makePsf(psfType=psfType[1], sigma=psf2, theta=theta2, psfSize=psfSize)
    centroids = np.column_stack((xposns + imSize[0]//2, yposns + imSize[1]//2, fluxes, fluxes2))
    return im1, im2, im1_psf, im2_psf, var_im1, var_im2, centroids, inds

