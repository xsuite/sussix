import numpy as np
import sussix as nafflib




#-----
# Henon map tune
Q_h = 0.2064898024701758
#-----
example_signals = {}
for x_start,label in zip([0.1,0.3,0.51],['low_J','mid_J','high_J']):
    example_signals[label] = nafflib.henon_map(x_start,0.35*x_start,Q_h,int(3e4))



def test_parse_real():

    for label,signal in example_signals.items():
        # Extracting signal
        x,px = signal
        z    = x - 1j*px
        N    = np.arange(len(z))

        # Choosing number of harmonics
        n_harm = 7

        # x-px lines
        # Take more here, since some might repeat
        spectrum_z = nafflib.harmonics(z,num_harmonics = 2*n_harm,window_order = 2,window_type = 'hann')
        r_z,_,_ = nafflib.find_linear_combinations(spectrum_z[1],fundamental_tunes= [spectrum_z[1][0]])

        # x-only lines
        spectrum_x = nafflib.harmonics(x,num_harmonics = n_harm,window_order = 2,window_type = 'hann')
        r_x,_,_ = nafflib.find_linear_combinations(spectrum_x[1],fundamental_tunes= [spectrum_x[1][0]])

        # Scanning x-lines and comparing with z-lines
        errors_Q = []
        errors_A = []
        for res,A,freq in zip(r_x,spectrum_x[0],spectrum_x[1]):
            spec_z_index = r_z.index(res)
            errors_Q.append(spectrum_z[1][spec_z_index]-freq)
            errors_A.append(np.abs(spectrum_z[0][spec_z_index])-np.abs(A))
        
        assert np.allclose(errors_Q,0,atol=1e-14,rtol=0), f'Q difference too large between x-only and x-px, for particle@{label}'
        assert np.allclose(errors_A,0,atol=1e-1,rtol=0),  f'|A| difference too large between x-only and x-px, for particle@{label}'



def test_x_px_handling():
    pass

def test_signal_generation():
    pass

def test_linear_combinations():
    pass