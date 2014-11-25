from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mp

import healpy as hp
import numpy as np
from qubic import (QubicAcquisition, read_spectra)
from myqubic import (create_sweeping_pointings, QubicAnalysis)

p_name = 'angspeed'
p_vals = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

nside = 256
racenter = 0.0
deccenter = -46.0

def oel(p_name, p_val, input_map):
    pointings = create_sweeping_pointings(parameter_to_change=p_name,
                                          value_of_parameter=p_val)
    print 'pointings created'

    # get the acquisition model = instrument model + pointing model
    band = 150
    acquisition = QubicAcquisition(band, pointings,
                                   kind='IQU',
                                   nside=nside)
    analysis = QubicAnalysis(acquisition,
                             input_map,
                             coverage_thr=0.2,
                             pickable=False,
                             noise=False,
                             run_analysis=False)
    o = analysis.Omega()
    e = analysis.Eta()
    l = analysis.Lambda(ndet=10)
    return o, e, l

npar = len(p_vals)
o = np.empty(npar)
e = np.empty(npar)
l = np.empty(npar)

spectra = read_spectra(0)
input_map = np.array(hp.synfast(spectra, nside)).T
for i, p_val in enumerate(p_vals):
    o[i], e[i], l[i] = oel(p_name, p_val, input_map)

mp.figure()
ax = mp.subplot(3, 1, 1, title='$\Omega$')
mp.plot(p_vals, o)
ax = mp.subplot(3, 1, 2, title='$\eta$', sharex=ax)
mp.plot(p_vals, e)
mp.subplot(3, 1, 3, title='$\lambda$', sharex=ax)
mp.plot(p_vals, l)
mp.savefig('scan_' + p_name + '_oel.png')

