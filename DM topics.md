Topics for Brown-Bag (Nov. 9, 2016)

1. PSF measurement -- issues and possible solutions?<br>
  * [Notebook](https://github.com/djreiss/diffimTests/blob/master/27.%20psf%20measurement%20evaluation%20-%20part%202.ipynb)
2. Implications for ZOGY?
  * [Notebook](https://github.com/djreiss/diffimTests/blob/master/26.%20algorithm%20shootout%20-%20detection-Copy6-dense.ipynb)
  * [Writeup](https://github.com/djreiss/diffimTests/blob/master/shootout.md)
  * Don't forget to show (and explain) S_corr
3. ZOGY in image space
  - What does it mean? [Paper](https://arxiv.org/pdf/1601.02655v2)
  - Why is it good?
     - Is it really necessary to do ZOGY in image space?
     - What if we do ZOGY in k-space on small stamps across image?
4. Problems with doing this stuff in image space
   * [Notebook](https://github.com/djreiss/diffimTests/blob/master/27.%20psf%20measurement%20evaluation%20-%20part%203.ipynb)
   * Related to issues I'm having with A&L pre-convolution + decorrelation (A&L+pc+dc) [notebook](https://github.com/djreiss/diffimTests/blob/master/27.%20psf%20measurement%20evaluation%20-%20try%20again%20to%20do%20decorr%2Bpreconv.ipynb)
   * Haven't tried padding PSFs for A&L+pc+dc
