import numpy as np
import pandas as pd

from .windowing import Hann
from .optimise import newton_method


def FFT_tune_estimate(z,n_forced = None):
    """
    Estimate the tune using an FFT. The signal is cropped to the closest power of 2 for improved accuracy.
    ----------------------------------------------------
        z        : complex array of the signal
        n_forced : number of turns to use for the FFT. if > len(z), the signal is padded with zeros
    ----------------------------------------------------
    """
    # Cropping signal to closest power of 2
    if n_forced is None:
        n_forced = 2**int(np.log2(len(z)))

    # Search for maximum in Fourier spectrum
    z_spectrum = np.fft.fft(z,n=n_forced)
    idx_max    = np.argmax(np.abs(z_spectrum))
    
    # Estimation of Tune with FFT
    tune_est   = idx_max/n_forced
    resolution = 1/n_forced

    return tune_est,resolution

def fundamental_frequency(x,px,Hann_order = 1):
    """
    Subroutine of the NAFF algorithm. 
    1. Applies a Hann window
    2. Estimates the fundamental frequency with an FFT 
    3. Refines the estimate with a complex Newton method
    4. Returns the main frequency and the amplitude.
    """

    # Windowing of the signal
    N   = np.arange(len(x))
    z   = np.array(x) - 1j*np.array(px)
    z_w = z * Hann(N, Nt=len(z),p=Hann_order)
    
    # Estimation of the tune with FFT
    tune_est,resolution = FFT_tune_estimate(z_w)

    # Preparing the estimate for the Newton refinement method
    if tune_est >= 0.5:
        tune_est = -(1.0 - tune_est)
    tune_est = tune_est - resolution

    # Refinement of the tune calulation
    tune,amplitude = newton_method(z_w,N,tune_est,resolution)

    return tune,amplitude


def NAFF(x,px,number_of_harmonics = 5,Hann_order = 1):
    """
    Applies the NAFF algorithm to find the spectral lines of a signal.
    """

    assert number_of_harmonics >=1, 'number_of_harmonics needs to be > 1'
    
    # Converting to numpy arrays
    x,px = np.array(x),np.array(px)

    # initialisation
    z  = x - 1j*px
    N  = np.arange(len(x))
    
    
    frequencies = []
    amplitudes  = [] 
    for _ in range(number_of_harmonics):

        # Computing frequency and amplitude
        freq,amp  = fundamental_frequency(x,px,Hann_order=Hann_order)

        # Saving results
        frequencies.append(freq)
        amplitudes.append(amp)

        # Substraction procedure
        zgs  = amp * np.exp(2 * np.pi * 1j * freq * N)
        z   -= zgs
        x,px = np.real(z), -np.imag(z)

    
    return pd.DataFrame({'amplitude':amplitudes,'frequency':frequencies})


def get_harmonics(x = None,px = None,y = None,py = None,zeta = None,pzeta = None,number_of_harmonics = 5,Hann_order = 1):
    """
    Computes the spectrum of a tracking data set for all canonical pairs provided
    """

    results = {}
    for pair in [(x,px,'x'),(y,py,'y'),(zeta,pzeta,'zeta')]:
        z,pz,plane = pair
        
        if z is not None:
            if pz is None:
                pz = np.zeros(len(z))

            # Computing spectral lines
            df = NAFF(z,pz, number_of_harmonics = number_of_harmonics,
                                Hann_order      = Hann_order)
            
            results[plane] = df

        else:
            results[plane] = None


    return results
