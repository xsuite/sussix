import numpy as np
import pandas as pd




def find_linear_combinations(frequencies,fundamental_tunes = [],max_jklm = 10):
    """
    Categorisation of resonances. Returns the linear combinations of the fundamental tunes that are closest to the provided frequencies.
    This should be called after get_harmonics to have a list of frequencies.
    """

    assert len(fundamental_tunes) == 3, "Need 3 fundamental tunes"

    # Create a 3D array of all possible combinations of j, k, l
    j,k,l,m = np.mgrid[-max_jklm:max_jklm+1, -max_jklm:max_jklm+1,-max_jklm:max_jklm+1,-max_jklm:max_jklm+1]

    # nu = j*Q_x + k*Q_y + l*Q_z + m
    all_combinations = j * fundamental_tunes[0] + k * fundamental_tunes[1] + l * fundamental_tunes[2] + m
    
    # Find the closest combination for each frequency
    jklm = []
    err = []
    for freq in frequencies:

        # Find the index of the closest combination
        closest_idx = np.unravel_index(np.argmin(np.abs(freq - all_combinations)), all_combinations.shape)

        # Get the corresponding values for l, j, k
        closest_combination = (j[closest_idx], k[closest_idx], l[closest_idx],m[closest_idx])
        closest_value = all_combinations[closest_idx]

        jklm.append(closest_combination)
        err.append(np.abs(closest_value-freq))

    return pd.DataFrame({'jklm':jklm,'err':err,'freq':frequencies})



def generate_signal(amplitudes,frequencies,N):
    """
    Generate a signal with the provided amplitudes and frequencies over turns N.
    """

    if isinstance(amplitudes,(float,int)):
        amplitudes = [amplitudes]
    if isinstance(frequencies,(float,int)):
        frequencies = [frequencies]

    assert len(amplitudes) == len(frequencies), "Amplitudes and frequencies must have the same length"

    signal = sum([A*np.exp(1j*(2*np.pi*(Q)*N  )) for A,Q in zip(amplitudes,frequencies)])
    x  =  signal.real
    px = -signal.imag

    return x,px