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
parser.add_option("-s", "--seed", dest="seed", default=0., help="Set random seed")
(options, args) = parser.parse_args()

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size
print 'I am {}/{}.'.format(rank, size)

nside = 256

racenter = 0.0
deccenter = -46.0

def run_ss(p_name, p_val, input_map):
    map_noI = copy(input_map)
    map_noI[:,0] *= 0

    pointings = create_sweeping_pointings(parameter_to_change=p_name,
                                          value_of_parameter=p_val)
    print 'pointings created'

    # get the acquisition model = instrument model + pointing model
    band = 150
    acquisition = QubicAcquisition(band, pointings,
                                   kind='IQU',
                                   nside=nside)
    analysis = QubicAnalysis(acquisition, input_map, coverage_thr=0.2, tol=1e-3, pickable=False, noise=False)
    coverage = analysis.coverage
    o = analysis.Omega()
    e = analysis.Eta()
    imap_nN = analysis.input_map_convolved
    omap_nN = analysis.reconstructed_map
    del analysis
    analysis_ns = QubicAnalysis(acquisition, input_map, coverage_thr=0.2, tol=1e-3, pickable=False, noise=True)
    omap_N = analysis_ns.reconstructed_map
    del analysis_ns
    analysis_QU = QubicAnalysis(acquisition, map_noI, coverage_thr=0.2, tol=1e-3, pickable=False, noise=False)
    imap_nI = analysis_QU.input_map_convolved
    omap_nI = analysis_QU.reconstructed_map
    del acquisition
    del analysis_QU
    del pointings
    ItoQU = np.empty((4, 3))
    Noise = np.empty((4, 3))
    QUmix = np.empty((4, 3))
    for i, cmin in enumerate(np.arange(0.2, 1.0, 0.2)):
        ItoQU[i] = RMS_of_map(coverage, imap_nN - omap_nN, cmin, cmin+0.2)
        Noise[i] = RMS_of_map(coverage, omap_N  - omap_nN, cmin, cmin+0.2)
        QUmix[i] = RMS_of_map(coverage, imap_nI - omap_nI, cmin, cmin+0.2)
    return ItoQU, Noise, QUmix, o, e #, imap_nN, omap_nN, omap_N, imap_nI, omap_nI, coverage

def RMS_of_map(cov, map, cov_min, cov_max):
    map_masked = map[:, ...][(cov_min < cov) * (cov <= cov_max)]
    return np.std(map_masked, axis=0)

p_name = options.param
p_val  = options.value
print p_name, "=", p_val

nrealizations = 1
ItoQU = np.zeros((nrealizations, 4, 3))
Noise = np.zeros((nrealizations, 4, 3))
QUmix = np.zeros((nrealizations, 4, 3))
for realization in xrange(nrealizations):
    seed = int(options.seed) + realization
    print seed
    np.random.seed(seed)
    spectra = read_spectra(0)
    input_map = np.array(hp.synfast(spectra, nside)).T
    ItoQU_, Noise_, QUmix_, o, e = run_ss(p_name, p_val, input_map)
    ItoQU[realization] = ItoQU_
    Noise[realization] = Noise_
    QUmix[realization] = QUmix_
dict = {p_name: p_val,
        'ItoQU': ItoQU,
        'Noise': Noise,
        'QUmix': QUmix,
        'Omega': o,
        'Eta': e}
if rank == 0:
    f_name = 'scan_ss_' + p_name + str(p_val) + 'seed' + str(options.seed) +'.pkl'
    with open(f_name, 'w') as f:
        dump(dict, f)
