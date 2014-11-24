from __future__ import division

import healpy as hp
import numpy as np

from pyoperators import MPI, pcg, DiagonalOperator, UnpackOperator
from pysimulators import SphericalEquatorial2GalacticOperator, Acquisition
from pysimulators.noises import _gaussian_psd_1f
from qubic import (equ2gal, map2tod, tod2map_all, tod2map_each, QubicInstrument, QubicAcquisition, read_spectra)
from qubic.io import write_map
from myqubic import (create_sweeping_pointings, QubicAnalysis)
from cPickle import dump
from copy import copy
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-p", "--param", dest="param", help="Chose name of a parameters to vary")
parser.add_option("-v", "--value", dest="value", help="Value of varied parameter")
(options, args) = parser.parse_args()

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
    analysis = QubicAnalysis(acquisition, input_map, coverage_thr=0.2, tol=1e-3, pickable=False, noise=False)
    o = analysis.Omega()
    e = analysis.Eta()
    l = analysis.Lambda(ndet=10)
    return o, e, l

p_name = options.param
p_val  = options.value
print p_name, "=", p_val

spectra = read_spectra(0)
input_map = np.array(hp.synfast(spectra, nside)).T
o, e, l = oel(p_name, p_val, input_map)
dict = {p_name: p_val,
        'Omega': o,
        'Eta': e,
        'Lambda': l}
if rank == 0:
    f_name = 'scan_ss_' + p_name + str(p_val) + '_oel.pkl'
    with open(f_name, 'w') as f:
        dump(dict, f)
