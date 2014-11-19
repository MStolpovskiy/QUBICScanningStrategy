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

def run_ss(pointings_params, input_map):
    map_noI = copy(input_map)
    map_noI[:,0] *= 0

    pointings = create_sweeping_pointings(
        pointings_params['radec'], 24, pointings_params['ts'],
        pointings_params['angspeed'], 
        pointings_params['delta_az'],
        pointings_params['nsweeps_el'],
        pointings_params['angspeed_psi'],
        pointings_params['maxpsi'], 
        delta_el_corr=pointings_params['delta_el_corr'],
        ss_az=pointings_params['az_s'], 
        ss_el=pointings_params['el_s'], 
        hwp_div=pointings_params['hwp'])
    print 'pointings created'

    # get the acquisition model = instrument model + pointing model
    band = 150
    acquisition = QubicAcquisition(band, pointings,
                                   kind='IQU',
                                   nside=nside)
    analysis = QubicAnalysis(acquisition, input_map, coverage_thr=0.2, tol=1e-3, pickable=False, noise=False)
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
ts = float(p_val) if (p_name == "ts") else 0.05
az_strategy =  p_val if (p_name == "az_strategy") else 'sss'
angspeed = float(p_val) if (p_name == "angspeed") else 1.
delta_az = float(p_val) if (p_name == "delta_az") else 30.
angspeed_psi = float(p_val) if (p_name == "angspeed_psi") else 1.
maxpsi = float(p_val) if (p_name == "maxpsi") else 15.
nsweeps_el = int(p_val) if (p_name == "nsweeps_el") else 300
delta_el_corr = 0. if (p_name == "delta_el_corr") else 0.
hwp = int(p_val) if (p_name == "hwp") else 8
pointings_params = {'radec': [racenter, deccenter],
                    'ts': ts,
                    'az_s': az_strategy,
                    'el_s': 'el_enlarged1',
                    'angspeed': angspeed,
                    'delta_az': delta_az,
                    'angspeed_psi': angspeed_psi,
                    'maxpsi': maxpsi,
                    'nsweeps_el': nsweeps_el,
                    'delta_el_corr': delta_el_corr,
                    'hwp': hwp}

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
    ItoQU_, Noise_, QUmix_, o, e = run_ss(pointings_params, input_map)
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
