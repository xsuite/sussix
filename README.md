# sussix

A Python implementation for the frequency analysis tool SUSSIX (R. Bartolini, F. Schmidt et al.), https://cds.cern.ch/record/702438/.

# Installation
```bash
pip install sussix
```

# Usage
Examples can be found in the `examples` folder.

Sussix uses phase space trajectories (x,px,y,py,zeta,pzeta) with the `get_harmonics()` function.

```python
# SUSSIX 
#-------
get_harmonics(  x     = None,
                px    = None,
                y     = None,
                py    = None,
                zeta  = None,
                pzeta = None,
                number_of_harmonics = 5,
                Hann_order          = 1)
#-------
# -> returns df where df['x'] has the complex amplitudes (amplitude + phase) and frequencies in the x plane
``` 

For each canonical pair, the spectrum of the data is computed using a NAFF approach and a well optimized solver to find the frequencies precisely. A Hann window is used to help with the convergence (see `sussix/windowing.py`) and the window order can be specified by the user. 

The indices (j,k,l,m) or the resonant frequencies can be found using `find_linear_combinations()` as done in the original SUSSIX code. 

The phase space trajectory can also be reconstructed from the frequency content using `generate_signal()`, which  simply sums the phasors optained from sussix:
```python
# Recontruction
sum([A*np.exp(1j*(2*np.pi*(Q)*N  )) for A,Q in zip(amplitudes,frequencies)])
```


