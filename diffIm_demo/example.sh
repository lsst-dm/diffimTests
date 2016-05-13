source $LSSTSW/setup.csh

cd obs_decam
git pull
setup -t b1823 -r .
scons install declare --tag=current
back

setup -t b1823 pipe_tasks
setup -k -t b1823 obs_decam

imageDifference.py calexpDir_b1631 --output decamDirTest --id visit=289820 ccdnum=11 --templateId visit=288976 --configfile diffimconfig.py --clobber-config


## Note to run with debugging, start ds9, then
setup -t 1822 display_ds9
addtopypath `pwd`  ## to add 'debug.py' to search path
## then command above, and add --debug to command line


# also to run with (lots of) debugging output:
imageDifference.py calexpDir_b1631 --output decamDirTest --id visit=289820 ccdnum=11 --templateId visit=288976 --configfile diffimconfig.py --clobber-config --debug -L debug >& debug.txt

# to run again (after running the first time) and don't re-do the image subtraction:
imageDifference.py calexpDir_b1631 --output decamDirTest --id visit=289820 ccdnum=11 --templateId visit=288976 --configfile diffimconfig.py --clobber-config --config doSubtract=False


# in ipython, load the output table and convert it into a pandas dataframe:
import lsst.daf.persistence as dp
butler=dp.Butler('decamDirTest')
sources=butler.get('deepDiff_diaSrc',visit=289820,ccdnum=11)
print sources[0].extract('ip_diffim_Naive*')
import pandas as pd
pd.DataFrame({col: sources.columns[col] for col in sources.schema.getNames()}).head()
