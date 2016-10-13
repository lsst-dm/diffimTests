# Notes on shoot-out between ZOGY and A&L (with decorrelation and optional pre-filtering).

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

All code and notebooks are in this repository, specifically the code is in the `diffimTests.py` module.

## Necessary ingredients for the shoot-out:

1. A function which can create simulated pairs of simple images (template and science images) with specificed numbers of point-sources of varying flux (one of which can be variable between the two images) and signal-to-noise, and with PSFs of varying shape. All PSFs are Gaussian, but they may be elongated in either axis, and rotated. PSFs may additionally be spatially varying, and their centroids may be offset between the two images.
2. A "clean-room" pure-python method that can apply A&L image subtraction to these simulated images, including both pre-filtering and decorrelation.
3. A "clean-room" pure python method that can apply the ZOGY algorithm to these simulated images. It can do so either completely in Fourier space, or by only computing kernels in Fourier space and then convolving the images in real (image) space. It can also compute the ZOGY matched-filtered image $S$ and its corrected version $S_{corr}$.
4. A wrapper that performs the LSST-stack (`ip_diffim`) version of A&L on the simulated images, with pre-filtering (pre-convolution of the science image with its PSF) *or* decorrelation. (Unlike the clean-room implementation described above, currently for this one, decorrelation with pre-filtering does not work.)

A comparison of the three different implementations (actually, four) is shown in Cell[30] of [this notebook](https://github.com/djreiss/diffimTests/blob/master/25.%20Compare%20basic%20ZOGY%20and%20ALCZ%20with%20preconvolution-final.ipynb).

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

ZOGY is slightly ($\sim 50\%$) faster than A&L(stack) in these tests. It should be noted that A&L scales (primarily) with the number of bright sources selected for PSF matching, and partly with image size. ZOGY should scale only with image size. If the field is more crowded, it is likely that A&L run-time performance will worsen, whereas this is not true for ZOGY. For example, the same test on A&L (stack, no pre-filter)on simulated images with 10x as many soures has a timing of 43.2 sec. We should strongly consider whether to use the FT-based version of ZOGY if we choose to move forward with that algorithm as it is $\sim 3 - 8 \times$ faster than the real-space implementation.

## Performance

We will now investigate performance of the algorithms in terms of false positive/negative detections. We will describe the measures of performance below, and then define the different tests.

### Detection

We will use the rate of true-postive detections (fraction of input sources actually detected by the algorithm) and false-positive detections (number of false detections divided by number of input sources) to quantify the performance of each algorithm in each set of simulated images. Ideally, the true-positive rate would be 100%, and the false-positive rate would be 0%.

### Tests/simulations

For the simulated input images, we will vary:

* The number of static and variable sources
* The relative widths/shapes/offsets of PSFs between the two images. This will include: 
 - The "canonical" case of near-Gaussian PSFs with that of the science image being wider than that of the template.
 - The inverse case where the PSF of the science image is narrower (or equal to) that of the template
 - Cases where PSF of one image are elongated in one axis to be broader than that of the other image
* Cases where PSFs of science and/or template image are mis-measured by a certain amount (we will quantify it in terms of percentage mismeasurement of PSF sigma).
* Cases where images are systematically astrometrically offset from each other by a fraction of a pixel
* Cases where noise of one or both images are mis-measured.
* Combinations of all of the above.


