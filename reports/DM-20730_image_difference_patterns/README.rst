Image difference residual report notes
======================================

The merge of DM-19660 changed which visit combinations trigger
deconvolution and which ones are normal convolution.

* The ``old`` run is 2019-02-21.
* The ``recent`` run is 2019-05-21.

These repositories are located on Gabor's desktop as of 2019-07-19.

Old run figures
~~~~~~~~~~~~~~~

Figures: ``v412074_Screenshot_20190515_160748.png``,
``v410985_Screenshot_20190515_154400.png``

::

   repo_DM-17825/ingested/rerun/proc_2019-02-21/deepDiff/v410985/
   run_notes/2019-02/ap_pipe_DM-17825_cmd_2019-02-21.txt


Recent run
~~~~~~~~~~

Figure: ``Scr_v410985_recent.png``
::

   repo_2019-05-15/ingested/rerun/imgdiff_2019-05-21/deepDiff/v410985/
   ds9 -title v410985_recent diffexp-06.fits
   run_notes/2019-05/imgDiff_plain_vs_preconv_2019-05-21.txt

Figure: ``Scr_v412074_recent.png``
::

   repo_2019-05-15/ingested/rerun/imgdiff_2019-05-21/deepDiff/v412074/
   ds9 -title v412074_recent diffexp-06.fits
   run_notes/2019-05/imgDiff_plain_vs_preconv_2019-05-21.txt

Pre-convolution w/ recent pipeline
----------------------------------

Figures: ``Scr_v411269_preconv.png``,
``Scr_v412074_preconv.png``
::

   repo_2019-05-15/ingested/rerun/imgdiff_preconv_2019-05-21/deepDiff/v411269/
   ds9 -title v411269_preconv diffexp-06.fits
   run_notes/2019-05/imgDiff_plain_vs_preconv_2019-05-21.txt

Two calexps w/ recent pipeline
------------------------------

Figure:``Scr_v412074_410985_calexps.png``
::

   repo_2019-05-15/ingested/rerun/diff_412074_410985_calexps/deepDiff/v412074/
   ds9 -title v412074_410985 diffexp-06.fits 
   run_notes/2019-05/imgDiff_two_calexps_2019-05-21.txt

Figure:  ``Scr_v410985_412074_deconv_calexps.png``
::

   ingested/rerun/diff_410985_412074_deconv_calexps/deepDiff/
   ds9 -title v410985_412074_deconv v410985/diffexp-06.fits 
   run_notes/2019-05/imgDiff_two_calexps_2019-05-21.txt


