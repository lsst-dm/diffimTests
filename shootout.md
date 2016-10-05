# Notes on shoot-out between ZOGY and A&L (with decorrelation and optional pre-filtering).

All code and notebooks are in this repository, specifically the code is in the `diffimTests.py` module.

## Necessary ingredients for the shoot-out:

1. A function which can create simulated pairs of simple images (template and science images) with specificed numbers of point-sources of varying flux (one of which can be variable between the two images) and signal-to-noise, and with PSFs of varying shape. All PSFs are Gaussian, but they may be elongated in either axis, and rotated. PSFs may additionally be spatially varying, and their centroids may be offset between the two images.
2. A "clean-room" pure-python method that can apply A&L image subtraction to these simulated images, including both pre-filtering and decorrelation.
3. A "clean-room" pure python method that can apply the ZOGY algorithm to these simulated images. It can do so either completely in Fourier space, or by only computing kernels in Fourier space and then convolving the images in real (image) space. It can also compute the ZOGY matched-filtered image $S$ and its corrected version $S_{corr}$.
4. A wrapper that performs the LSST-stack (`ip_diffim`) version of A&L on the simulated images, with pre-filtering (pre-convolution of the science image with its PSF) *or* decorrelation. (Unlike the clean-room implementation described above, currently for this one, decorrelation with pre-filtering does not work.)

A comparison of the three different implementations (actually, four) is shown in Cell[30] of [this notebook](https://github.com/djreiss/diffimTests/blob/master/25.%20Compare%20basic%20ZOGY%20and%20ALCZ%20with%20preconvolution-Copy2.ipynb).

## Timings

I have compared the run-time of the algorithms on a pair of basic 512x512 -pixel images with 50 sources and an elongated PSF in the science image. For all A&L runs, decorrelation was enabled, and warping was disabled. For all tests, PSFs were 25x25 pixels. The comparisons are in [this notebook](https://github.com/djreiss/diffimTests/blob/master/25.%20Compare%20basic%20ZOGY%20and%20ALCZ%20with%20preconvolution-Copy2.ipynb). Results:

| Method        | Pre-filtering enabled? | Time to execute (ms) |
|---------------|------------------------|----------------------|
| A&L (custom)  | Yes | 11,800 |
| A&L (custom)  | No  | 9,910 |
| A&L (stack)   | Yes | 4,360  |
| A&L (stack)   | No  | 3,890  |
| ZOGY (real)   | -   | 584    |
| ZOGY (FT)     | -   | 359    |

ZOGY is $\sim 6\times$ faster than the optimized version of A&L in the LSST stack.
