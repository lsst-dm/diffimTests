# use instructions from Simon:
# https://community.lsst.org/t/running-processfile-with-lsst-stack-v12-0/904/19?u=reiss

git clone https://github.com/lsst/pipe_tasks
cd pipe_tasks
git checkout tickets/DM-6924
setup -r . -t b2063
scons lib python
back

git clone https://github.com/simonkrughoff/obs_file
cd obs_file
git checkout tickets/DM-6924
setup -r . -t b2063
scons
back

processCcd.py test_out/ --id filename=im1.fits --output test_out --config charImage.installSimplePsf.fwhm=1.0 --config charImage.repair.doCosmicRay=False --clobber-config --configfile ./processCcdConfig.py --config isr.noise=300 isr.addNoise=True
processCcd.py test_out/ --id filename=im2.fits --output test_out --config charImage.installSimplePsf.fwhm=1.0 --config charImage.repair.doCosmicRay=False --clobber-config --configfile ./processCcdConfig.py --config isr.noise=300 isr.addNoise=True

#processFile.py im1.fits --outputCatalog cat1.fits --outputCalexp calexp1.fits
#processFile.py im2.fits --outputCatalog cat2.fits --outputCalexp calexp2.fits

imageDifference.py test_out --id fileroot=im2 --templateId fileroot=im1 --output ./test_dr1 --configfile diffimConfig.py --clobber-config --clobber-versions
