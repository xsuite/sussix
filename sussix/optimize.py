import numpy as np
import numba



@numba.jit(nopython=True)
def raise2powerArray(a,b):
    out    = np.zeros(b) + 1j*np.zeros(b)
    out[0] = 1
    out[1] = a
    for i in range(1,len(out)):
        out[i] = out[i-1]*out[1]
    return out


def laskar_dfft(freq,N,z):
    """
    Discrete fourier transform of z, as defined in A. Wolski, Sec. 11.5.
    In a typical dfft , freq = m/Nt where m is an integer. Here m could take any value.
    Note: this will differ from sussix.f.calcr by a factor 1/Nt (but does not slow the convergence of the Newton method, so not a problem)
    ----------------------------------------------------
        freq: discrete frequency to evaluate the dfft at
        N   : turn numbers of the signal
        z   : complex array of the signal

        returns dfft and its derivative.
    ----------------------------------------------------
    """
    Nt = len(z)

    # Argument of the summation
    # raise2power is used to save computation time, eq. to np.exp(-2*np.pi*1j*freq*N)
    to_sum = 1/Nt*raise2powerArray(np.exp(-2*np.pi*1j*freq),len(N))*z

    # Derivative factor
    deriv_factor = 1j*N

    # dfft and its derivative
    _dfft            = np.sum(to_sum)
    _dfft_derivative = np.sum(deriv_factor*to_sum)
    return _dfft,_dfft_derivative


def newton_method(z,N,freq_estimate,resolution,tol = 1e-10,num_macro_iterations = 10,num_micro_iterations = 100):


    # Increase resolution by factor 5
    resolution = resolution/5  
    #---------------------------------------


    # Initialisation of the Newton method
    #---------------------------------------
    root1 = freq_estimate
    freq_found = []
    amp_found  = []
    #---------------------------------------


    # Start the Newton method
    #========================
    dfft,dfft_d = laskar_dfft(root1,N,z)    
    droot1 = dfft.real*dfft_d.real + dfft.imag*dfft_d.imag

    root2 = 0
    droot2 = 0
    
    for _ in range(num_macro_iterations):
        root2 = root1+resolution

        dfft,dfft_d  = laskar_dfft(root2,N,z)
        droot2 = dfft.real*dfft_d.real + dfft.imag*dfft_d.imag

        
        if (droot1 <= 0) and (droot2 >= 0):
            freq1, freq2, dfreq1, dfreq2 = root1, root2, droot1, droot2

            
            for __ in range(num_micro_iterations):
                ratio = -dfreq1 / dfreq2 if abs(dfreq2) > 0 else 0.0

                freq3 = (freq1 + ratio * freq2) / (1.0 + ratio)

                
                dfft,dfft_d = laskar_dfft(freq3,N,z)
                dfreq3 = dfft.real*dfft_d.real + dfft.imag*dfft_d.imag


                if dfreq3 <= 0.0:
                    if freq1 == freq3:
                        break
                    freq1, dfreq1 = freq3, dfreq3
                else:
                    if freq2 == freq3:
                        break
                    freq2, dfreq2 = freq3, dfreq3

                if abs(freq2 - freq1) <= tol:
                    break

            
            freq_found.append(freq3)
            amp_found.append(np.abs(dfft))
            
        root1, droot1 = root2, droot2

    if len(amp_found) == 0:
        # No frequency found! (For accelerator physics, should never happen outside of chaotic layers)
        return np.nan + 1j*np.nan, np.nan
    else:
        idx_max = np.argmax(amp_found)
        frequency   = freq_found[idx_max]
        amplitude,_ = laskar_dfft(frequency,N,z)
        return amplitude,frequency
