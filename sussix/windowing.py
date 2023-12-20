import numpy as np
import math



def Hann(N,Nt = None,p=1):
    """
    Hann's window, centered at the middle of the dataset.
    ----------------------------------------------------
        N : index of the turn
        Nt: total number of turns
    ----------------------------------------------------
    """
    if Nt is None:
        Nt = np.max(N)
    if np.mod(Nt,2) == 0:
        center = Nt//2 - 1
    else:
        center = Nt//2
    return (2**p)*math.factorial(p)**2/(math.factorial(2*p)) * (1+np.cos(2*np.pi*(N-center)/Nt))**(p)

