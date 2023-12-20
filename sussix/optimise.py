import numpy as np
import numba



@numba.jit(nopython=True)
def raise2power(my_exp,N):
    out    = np.zeros(len(N)) + 1j*np.zeros(len(N))
    out[0] = 1
    out[1] = my_exp
    for i in range(1,len(out)):
        out[i] = out[i-1]*out[1]
    return out


def Laskar_DFFT(freq,N,z):
    """
    Discrete fourier transform of z, as defined in A. Wolski, Sec. 11.5.
    In a typical DFFT , freq = m/Nt where m is an integer. Here m could take any value.
    Note: this will differ from sussix.f.calcr by a factor 1/Nt (but does not slow the convergence of the Newton method, so not a problem)
    ----------------------------------------------------
        freq: discrete frequency to evaluate the DFFT at
        N   : turn numbers of the signal
        z   : complex array of the signal

        returns DFFT and its derivative.
    ----------------------------------------------------
    """
    Nt = len(z)

    # Argument of the summation
    # raise2power is used to save computation time, eq. to np.exp(-2*np.pi*1j*freq*N)
    to_sum = 1/Nt*raise2power(np.exp(-2*np.pi*1j*freq),N)*z

    # Derivative factor
    deriv_factor = 1j*N

    _DFFT            = np.sum(to_sum)
    _DFFT_derivative = np.sum(deriv_factor*to_sum)
    return _DFFT,_DFFT_derivative


def newton_method(z,N,tune_est,resolution,tol = 1e-10):

    # Initialisation using the fortran names
    tune_test = np.zeros(10)
    tune_val  = np.zeros(10)

    tunea1 = tune_est
    deltat = resolution
    err    = tol
    num    = 0

    # Increase resolution by factor 5
    deltat /= 5


    # Start the Newton method
    DFFT,DFFT_d = Laskar_DFFT(tunea1,N,z)    
    dtunea1 = DFFT.real*DFFT_d.real + DFFT.imag*DFFT_d.imag

    tunea2 = 0
    dtunea2 = 0
    
    for ntest in range(1, 11):
        tunea2 = tunea1+deltat

        DFFT,DFFT_d  = Laskar_DFFT(tunea2,N,z)
        dtunea2 = DFFT.real*DFFT_d.real + DFFT.imag*DFFT_d.imag

        
        if (dtunea1 <= 0) and (dtunea2 >= 0):
            tune1, tune2, dtune1, dtune2 = tunea1, tunea2, dtunea1, dtunea2

            
            for ncont in range(1, 101):
                ratio = -dtune1 / dtune2 if abs(dtune2) > 0 else 0.0

                tune3 = (tune1 + ratio * tune2) / (1.0 + ratio)

                
                DFFT,DFFT_d = Laskar_DFFT(tune3,N,z)
                dtune3 = DFFT.real*DFFT_d.real + DFFT.imag*DFFT_d.imag


                if dtune3 <= 0.0:
                    if tune1 == tune3:
                        break
                    tune1, dtune1 = tune3, dtune3
                else:
                    if tune2 == tune3:
                        break
                    tune2, dtune2 = tune3, dtune3

                if abs(tune2 - tune1) <= err:
                    break

            
            tune_test[num] = tune3
            tune_val[num]  = np.abs(DFFT)
            num += 1
            


        tunea1, dtunea1 = tunea2, dtunea2

    idx_max = np.argmax(tune_val[:num])
    tune      = tune_test[idx_max]
    amplitude,_ = Laskar_DFFT(tune,N,z)
    return tune,amplitude
