import numpy as np




def parse_real_signal(amplitudes,frequencies,conjugate_tol=1e-10,to_pandas = False):
    
    A,Q = amplitudes,frequencies
    phasors = A*np.exp(2*np.pi*1j*Q)

    freq = []
    amp  = []

    for i in range(len(Q)-1):

        # Finding closest conjugate pair
        comparisons = phasors[i] + phasors
        pair_idx = np.argmin(np.abs(np.imag(comparisons)))
        pair_A = np.array([A[i],A[pair_idx]])
        pair_Q = np.array([Q[i],Q[pair_idx]])

        # Check if the pair is a complex conjugate, otherwise both freqs are kept
        if np.abs(np.diff(np.abs(pair_Q)))[0]>conjugate_tol:
            freq.append(Q[i])
            amp.append(A[i])
            continue

        # Creating avg amplitude and freq
        real = np.mean(np.real(pair_A))
        imag = np.mean(np.abs(np.imag(pair_A)))
        sign = np.sign(np.imag(pair_A))[pair_Q>=0]

        if pair_Q[0]==pair_Q[1]:
            # the pair is a copy of itself (DC signal case)
            freq.append(pair_Q[0])
            amp.append(real+1j*imag)
        else:
            # Complex conjugate found, adding only once
            if np.mean(np.abs(pair_Q)) not in freq:
                freq.append(np.mean(np.abs(pair_Q)))
                amp.append(2*(real+sign[0]*1j*imag))
    
    if to_pandas:
        import pandas as pd
        return pd.DataFrame({'amplitude':amp,'frequency':freq})
    else:
        return np.array(amp),np.array(freq)
    


def find_linear_combinations(frequencies,fundamental_tunes = [],max_jklm = 10,to_pandas = False):
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

    if to_pandas:
        import pandas as pd
        return pd.DataFrame({'jklm':jklm,'err':err,'freq':frequencies})
    else:
        return np.array(jklm),np.array(err),np.array(frequencies)



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


def generate_pure_KAM(amplitudes,jklm,fundamental_tunes,N,return_frequencies=False):
    """
    Generate a signal with the provided amplitudes and frequencies over turns N.
    """

    if isinstance(amplitudes,(float,int)):
        amplitudes = [amplitudes]
    if isinstance(jklm,(float,int)):
        jklm = [jklm]

    assert len(amplitudes) == len(jklm), "amplitudes and jklm must have the same length"

    # Generating the signal
    Q1,Q2,Q3 = fundamental_tunes
    signal = sum([A*np.exp(2 * np.pi * 1j * (j*Q1 + k*Q2 + l*Q3 + m) * N) for A,(j,k,l,m) in zip(amplitudes,jklm)])
    x  =  signal.real
    px = -signal.imag

    if return_frequencies:
        return x,px,[j*Q1 + k*Q2 + l*Q3 + m for (j,k,l,m) in jklm]
    else:
        return x,px