import lsst.daf.persistence as dp
butler=dp.Butler('decamDirTest')
sources=butler.get('deepDiff_diaSrc',visit=289820,ccdnum=11)
print sources[0].extract('ip_diffim_Naive*')
import pandas as pd
df = pd.DataFrame({col: sources.columns[col] for col in sources.schema.getNames()})

