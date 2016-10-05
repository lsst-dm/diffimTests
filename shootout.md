# Notes on shootout between ZOGY and A&L (with decorrelation and optionally pre-filtering).

1. I have a simple function which can create simulated pairs of images (template and science images) with specificed numbers of point-sources of varying flux and signal-to-noise, and with PSFs of varying shape. All PSFs are Gaussian, but they may be elongated in either axis, and rotated.
2. I have implemented a "clean-room" pure-python method that can apply A&L image subtraction to these simulated images, including both pre-filtering and decorrelation in the `diffimTests.py` module in this repository.
3. I have also implemented a "clean-room" method that can apply the ZOGY algorithm to these simulated images. It can do so either completely in Fourier space, or by only computing kernels in Fourier space and then convolving the images in real (image) space. It can also compute the ZOGY matched-filtered image $S$ and its corrected version $S_{corr}$.
4. I have implemented a wrapper that performs the LSST-stack (`ip_diffim`) version of A&L on the simulated images, with pre-filtering (pre-convolution of the science image with its PSF) *or* decorrelation. (Unlike my clean-room implementation described above, currently for this one, decorrelation with pre-filtering does not work.)

## Timings

I have compared the run-time of the algorithms on a pair of basic 512x512 -pixel images. For the A&L runs, decorrelation was enabled. The comparisons are in [this notebook](https://github.com/djreiss/diffimTests/blob/master/25.%20Compare%20basic%20ZOGY%20and%20ALCZ%20with%20preconvolution-Copy2.ipynb). Results:

| Method        | Time to execute (ms) |
|---------------|---|
| A&L (custom)  | 11,800 |
| A&L (stack)   | 2,890  |
| ZOGY (real)   | 586    |
| ZOGY (FT)     | 373    |
 