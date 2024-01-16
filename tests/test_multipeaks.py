import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy.stats as sciStat
import time

import sussix

# Creating dummy signal
#==============================================
N  = np.arange(int(1e5))
Q0 = 0.31025793875089835
Qs = 0.002
dQ = Qs/12
Jx = (0.5*(10**2))

n_bands_Qs = 1
n_bands_dQ = 5
i,j    = np.arange(-n_bands_dQ,n_bands_dQ+1),np.arange(-n_bands_Qs,n_bands_Qs+1)
Ai,Aj  = sciStat.cauchy.pdf(i/np.max(i),0,0.05),sciStat.cauchy.pdf(j/np.max(j),0,0.05)
Ai,Aj  = Ai/np.max(Ai),Aj/np.max(Aj)

amplitudes  = np.array([[ (np.sqrt(2*Jx)*_Ai*_Aj) for _i,_Ai in zip(i,Ai) ] for _j,_Aj in zip(j,Aj) ]).flatten()
frequencies = np.array([[Q0+ _j*Qs + _i*dQ for _i,_Ai in zip(i,Ai) ] for _j,_Aj in zip(j,Aj) ]).flatten() 

expected = pd.DataFrame({'amplitude':amplitudes,'frequency':frequencies})
expected.sort_values(by='amplitude',ascending=False,inplace=True)
expected.reset_index(drop=True,inplace=True)

x,px    = sussix.analysis.generate_signal(expected.amplitude,expected.frequency,N)
#==============================================

# Running sussix
#==============================================
results = sussix.get_harmonics( x       = x, 
                                px      = px,
                                y       = x,
                                py      = px,
                                zeta    = x,
                                pzeta   = px,
                                number_of_harmonics = len(expected),Hann_order = 1)
#==============================================

# Comparing results
#==============================================
assert len(results['x']) == len(expected), "Number of harmonics is not the same"
assert (results['x'] == results['y']).all().all()   , "x, y and zeta should be the same"
assert (results['x'] == results['zeta']).all().all()   , "x, y and zeta should be the same"
assert np.allclose(results['x'].sort_values(by='frequency').frequency.values    ,expected.sort_values(by='frequency').frequency.values,atol=1e-9,rtol=0), "Expected tolerance not met"
assert np.allclose(results['y'].sort_values(by='frequency').frequency.values    ,expected.sort_values(by='frequency').frequency.values,atol=1e-9,rtol=0), "Expected tolerance not met"
assert np.allclose(results['zeta'].sort_values(by='frequency').frequency.values ,expected.sort_values(by='frequency').frequency.values,atol=1e-9,rtol=0), "Expected tolerance not met"
#==============================================