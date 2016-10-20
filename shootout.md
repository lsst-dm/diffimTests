# Notes on shoot-out between ZOGY and A&L (with decorrelation and optional pre-filtering).

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

All code and notebooks are in this repository, specifically the code is in the `diffimTests.py` module.

## Necessary ingredients for the shoot-out:

1. A function which can create simulated pairs of simple images (template and science images) with specificed numbers of point-sources of varying flux (one of which can be variable between the two images) and signal-to-noise, and with PSFs of varying shape. All PSFs are Gaussian, but they may be elongated in either axis, and rotated. PSFs may additionally be spatially varying, and their centroids may be offset between the two images.
2. A "clean-room" pure-python method that can apply A&L image subtraction to these simulated images, including pre-filtering *or* decorrelation (see *Note* below).
3. A "clean-room" pure python method that can apply the ZOGY algorithm to these simulated images. It can do so either completely in Fourier space, or by only computing kernels in Fourier space and then convolving the images in real (image) space. It can also compute the ZOGY matched-filtered image $S$ and its corrected version $S_{corr}$.
4. A wrapper that performs the LSST-stack (`ip_diffim`) version of A&L on the simulated images, with pre-filtering (pre-convolution of the science image with its PSF) *or* decorrelation (see *Note* below).

*Note: Currently for both implementations of A&L, decorrelation with pre-filtering does not work. [UPDATE: It appears that I have figured out how to get it to work, with creative usage of `fftshift()` and `ifftshift()` in various locations. Still needs to be thoroughly vetted.*

A comparison of the three different implementations (actually, four) is shown in Cell[30] of [this notebook](https://github.com/djreiss/diffimTests/blob/master/25.%20Compare%20basic%20ZOGY%20and%20ALCZ%20with%20preconvolution-final.ipynb), and in the figure below, for an example when the template's PSF is wider than that of the science image. Both A&L image differences are computed with the decorrelation "afterburner" enabled.

![](shootout/fig0.png)
We clearly see the deconvolution artifacts surrounding locations of bright stars in the A&L case with no pre-convolution, however with pre-convolution enabled, the resulting likelihood image shows only insignificant artifacts surrounding those locations.

## Observations about the different methods

1. I note that the combination pre-filtering + A&L + decorrelation has the potential to lose pixels around edges and masks due to up to *three* convolutions. In contrast ZOGY (with convolutions in image space) has effectively *one* convolution of each image, and should lose fewer pixels.
2. A&L loses sensitivity if a large number of sources are increasing in flux between the two simulated images. This is because it adjusts the kernel to scale the fluxes given the assumption that no sources are changing. This is a not a big concern in practice.
3. A&L (stack version) run-time scales with number of sources, as it performs PSF matching surrounding the bright stars only. For example, the timings in the table below are for images with 50 sources. If we increase the number to 250, the timing for A&L (stack; pre-filtering=No) is 2.31s, an increase of 25%.
4. ZOGY provides a method for correcting the corrected likelihood image $S_{corr}$ by pixel-wise variance, as well as by astrometric errors between the two images (which must be measured). I have implemented both of these corrections (the correction for astrometric noise uses ZOGY eqns. 30-33), and note that performing them requires the an additional two convolutions of each of the science and template images (not a factor regarding loss of border pixels, but just timing).

## Timings

I have compared the run-time of the algorithms on a pair of basic 2k x 2k -pixel images with 250 sources and a slightly elongated PSF in the science image. For all A&L runs, decorrelation was enabled, and warping was disabled. For all tests, PSFs were 25x25 pixels. Timings were performed using `%timeit` in an IPython notebook, and using a single core on `lsst-dev` (Intel Xeon E5-2687W @ 3.10GHz). The ZOGY "pre-filtering enabled" runs are actually those where $S_{corr}$ is computed in addition to $D$ (as pre-filtering is not necessary with ZOGY, but the resulting $S_{corr}$ corresponds to the match-filtered $D$, as it does for A&L with pre-filtering enabled). The comparisons are in [this notebook](https://github.com/djreiss/diffimTests/blob/master/25.%20Compare%20basic%20ZOGY%20and%20ALCZ%20with%20preconvolution-final.ipynb). Results:

| Method        | Pre-filtering? | Time (sec.) |
|---------------|------------------------|----------------------|
| A&L (custom)  | Yes | 487   |
| A&L (custom)  | No  | 442   |
| A&L (stack)   | Yes | 30.7  |
| A&L (stack)   | No  | 26.1  |
| ZOGY (real)   | Yes | 21.2   |
| ZOGY (real)   | No  | 14.8   |
| ZOGY (FT)     | Yes | 8.69   |
| ZOGY (FT)     | No  | 2.3   |

ZOGY is slightly ($\sim 2\times$) faster than A&L(stack) in these tests. It should be noted that A&L scales (primarily) with the number of bright sources selected for PSF matching, and partly with image size. ZOGY should scale only with image size. If the field is more crowded, it is likely that A&L run-time performance will worsen, whereas this is not true for ZOGY. For example, the same test on A&L (stack, no pre-filter) on simulated images with 10x as many soures has a timing of 43.2 sec., or nearly $2\times$ slower. We should strongly consider whether to use the FT-based version of ZOGY if we choose to move forward with that algorithm as it is $\sim 3 - 8 \times$ faster than the real-space implementation.

## Performance

We will now investigate performance of the algorithms in terms of false positive/negative detections. We will describe the measures of performance below, and then define the different tests.

### Detection

We will use the rate of true-postive detections (fraction of input sources actually detected by the algorithm) and false-positive detections (number of false detections divided by number of input sources) to quantify the performance of each algorithm in each set of simulated images. Ideally, the true-positive rate would be 100%, and the false-positive rate would be 0%.

### Tests/simulations

For the simulated input images, we will vary:

1. The number of static and variable sources

2. The relative widths/shapes/offsets of PSFs between the two images. This will include: 
 
 2a. The "canonical" case of near-Gaussian PSFs with that of the science image being wider than that of the template.
 
 2b. The inverse case where the PSF of the science image is narrower (or equal to) that of the template
 
 2c. Cases where PSF of one image are elongated in one axis to be broader than that of the other image
 
3. Cases where PSFs of science and/or template image are mis-measured by a certain amount (we will quantify it in terms of percentage mismeasurement of PSF sigma).
 
4. Cases where images are systematically astrometrically offset from each other by a fraction of a pixel

5. Cases where noise of one or both images are mis-measured.

6. Combinations of all of the above.

## First results

### Baseline

To establish a baseline, we performed an initial set of simulations described by (2a.) above in [this notebook](https://github.com/djreiss/diffimTests/blob/master/26.%20algorithm%20shootout%20-%20detection-Copy2.ipynb), using elliptical Gaussian PSFs of widths [1.6, 1.6] pixels (template) and [1.8, 2.2] pixels (science). The science image PSF was also rotated by 45 degrees. We inserted 100 static sources in both images and 10 "new" sources in the science image with a SNR of 5-sigma. Each simulation was performed 100 times with different source locations and (static) source brightnesses. The results are summarized below:

![](shootout/fig1.png)

Here we show the distributions of true-positives (left) and false-positives (right) for two versions of each algorithm. `ALstack` is the currently-implemented A&L algorithm in the LSST software stack, including diffim decorrelation ([DMTN-021](http://dmtn-021.lsst.io/)), and `ALstack_noDecorr` is the same with decorrelation disabled. `ZOGY` is the standard ZOGY algorithm, implemented in real-space. `SZOGY` denotes the computation of the ZOGY corrected likelihood image $S_{corr}$. In all cases, detection was performed at $5-\sigma$ and in all cases except `SZOGY` detection included the pre-matched-filter convolution step. It is apparent that the `SZOGY` method seems to slightly outperform in both true positives (with a slightly greater percentage of the ten variable sources detected) and false positives (with a slightly greater weighting of zero false positives detected). We obtained the same result when implanting fake "new" sources of both 3.3-sigma and 10-sigma SNR. This is not entirely surprising since we are supplying ZOGY with additional information (exactly correct PSFs) that A&L does not utilize.

### Astrometric mis-registration
We performed an initial set of simulations to test case (4.) above; i.e. when the two images are systematically mis-aligned by 0.3 pixels in both x- and y- directions. A&L should be able to account for this mis-alignment by incorporating it into the PSF matching kernel. ZOGY handles this situation by incorporating a measured astrometric "noise" into the variance which goes into the denominator of $S_{corr}$ (basically reducing the dynamic range for detection in regions surrounding bright stars). The results of these simulations are in [this notebook](https://github.com/djreiss/diffimTests/blob/master/26.%20algorithm%20shootout%20-%20detection-Copy3.ipynb), and shown below for 5-sigma "new" sources:

![](shootout/fig2.png)
Clearly, the `ZOGY` uncorrected diffim cannot handle the astrometric offsets, but the corrected `SZOGY` likelihood image performs similarly to the case (above) with no astrometric offset. There is, however a slight increase in false-positives for the `SZOGY` case relative to A&L. Surprisingly, A&L seems to improve in performance in this case relative to the baseline above. This is currently not understood.

### PSF mismeasurement
We also performed an initial set of simulations to test case (3.) above; i.e. when the PSF is mis-measured, at what point does this lead to "breakage" of ZOGY? We used the same parameters as above, except in each case, we reported a slightly incorrect science image PSF (both FWHMs and rotation angle). The results are in [this notebook](https://github.com/djreiss/diffimTests/blob/master/26.%20algorithm%20shootout%20-%20detection-Copy5.ipynb) and are summarized below:

![](shootout/fig3.png)
This complicated set of plots shows true- and false-positives for `ALstack` and `SZOGY` (as described above) for gradually increasing mis-measurement of the science image PSF. We plot the PSF mismeasurement error in units of RMS(truePSF - falsePSF)*100. In these units, ($\sim 0.04$) along the x-axis of these plots corresponds to a $\sim 5\%$ mis-measurement of the PSF FWHM. Unsurprisingly, the PSF mis-measurement has no effect on A&L (as this information is not used by the algorithm). It also has little effect on the rate of true-positive detection for `SZOGY`. However, the rate of false-positive detections clearly explodes right around $\sim 0.04$, or about 5\% mis-measurement of the PSF FWHM. This might correspond to an unrealistically pessimistic PSF measurement error (hopefully we can do better, even in crowded or very sparse fields!), but this essentially sets the limit to which ZOGY can be expected to perform adequately relative to A&L.

Of note, it was pointed out (by Eran Ofek) that these mis-measured-PSF simulations included no astrometric "scintillation", which, if included, will decrease A&L sensitivity but should not affect PSF measurement (and therefore not affect ZOGY). According to the LSST Science Book, we don't expect scintillation to be an issue at greater than $\sim 5$mas, or 0.025 pixels (even less for coadds of two 15-sec. LSST snaps). I performed these simulations in [this notebook](https://github.com/djreiss/diffimTests/blob/master/26.%20algorithm%20shootout%20-%20detection-Copy6.ipynb) and there appears to be no discernable effect.

