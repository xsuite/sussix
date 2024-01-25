
import numpy as np
from .naff import harmonics,tune


def get_tune(x, order=2, interpolation = 0):
    assert interpolation == 0, "DEPRECATED: interpolation is no longer supported"

    return float(tune(x , px=None, window_order=order, window_type='hann'))



def get_tunes(x, N = 1, order=2, interpolation = 0):
    assert interpolation == 0, "DEPRECATED: interpolation is no longer supported"


    if np.all(np.imag(np.asarray(x)) == 0):
        N *= 2

    amplitudes,frequencies = harmonics(x,px = 0,    num_harmonics = N,
                                                    window_order = order,
                                                    window_type = 'hann',
                                                    to_pandas = False)
    

    A = amplitudes[frequencies>=0]
    B = amplitudes[frequencies<0]
    Q = frequencies[frequencies>=0]


    return Q, A, B
    

def get_tunes_all(x, N=1, order=2, interpolation=0):
    assert interpolation == 0, "DEPRECATED: interpolation is no longer supported"

    amplitudes,frequencies = harmonics(x,px = 0,    num_harmonics = N,
                                                    window_order = order,
                                                    window_type = 'hann',
                                                    to_pandas = False)
    Q = frequencies
    A = amplitudes
    return Q, A



# def multiparticle_tunes(x, order=2, interpolation=0):   
#     assert interpolation == 0, "DEPRECATED: interpolation is no longer supported"

#     q_i = np.empty_like(x[:,0], dtype=np.float64)
#     for ii in range(len(x)):
#         q_i[ii] = get_tune(x[ii], order, interpolation)
#     return q_i
