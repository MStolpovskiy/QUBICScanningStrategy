from __future__ import division

from myqubic import create_sweeping_pointings
from qubic import (QubicAcquisition,
                   PlanckAcquisition,
                   QubicPlanckAcquisition)
from qubic.io import write_map
import gc
import healpy as hp
import numpy as np
from pyoperators import MPI, pcg
import ConfigParser
import json
from itertools import product
import os

debug_mode = True
maxiter = 300

config = ConfigParser.RawConfigParser()
config.read('angspeed-delta_az.cfg')

angspeed_ar = json.loads(config.get('Scanning_Strategy', 'angspeed'))
delta_az_ar = json.loads(config.get('Scanning_Strategy', 'delta_az'))

time_on_const_elevation = config.getint('Scanning_Strategy', 'time_on_const_elevation')
angspeed_psi = config.getfloat('Scanning_Strategy', 'angspeed_psi')
maxpsi = config.getfloat('Scanning_Strategy', 'maxpsi')
hwp_div = config.getint('Scanning_Strategy', 'hwp_div')
dead_time = config.getfloat('Scanning_Strategy', 'dead_time')

sampling_period = config.getfloat('Observation', 'sampling_period')

nep = config.getfloat('Observation', 'nep')
nep_normalization = {'1year': 365
    } [config.get('Observation', 'nep_normalization')] * 86400
fknee = config.getfloat('Observation', 'fknee')

nside = config.getint('Analysis', 'nside')
nrealizations = config.getint('Analysis', 'nrealizations')

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

print 'I am {} of {}'.format(rank, size)
MPI.COMM_WORLD.Barrier()

duration_effective = nep_normalization / 3600  # in hours
duration_actual = 24                           # in hours
nep_normalization = np.sqrt(duration_effective / duration_actual)
detector_nep = nep / nep_normalization

if rank == 0:
    path = './'
    directory = path + 'angspeed-delta_az_scan/'
    if debug_mode: directory += 'debug_mode/'
    directory += 'const_el_time{}_angspeed_psi{}_dead_time{}_fknee{}_nside{}'.format(time_on_const_elevation,
                                                                                     angspeed_psi,
                                                                                     dead_time,
                                                                                     fknee,
                                                                                     nside)
    if not os.path.exists(directory):
        os.makedirs(directory)

for (angspeed, delta_az) in product(angspeed_ar, delta_az_ar):
    print 'Angspeed, delta_az = {}, {} (rank={})'.format(angspeed, delta_az, rank)
    nsw = int(time_on_const_elevation / (delta_az / angspeed))
    pointings = create_sweeping_pointings(duration=duration_actual,
                                          sampling_period=sampling_period,
                                          angspeed=angspeed,
                                          delta_az=delta_az,
                                          nsweeps_per_elevation=nsw,
                                          ss_psi='sss',
                                          angspeed_psi=angspeed_psi,
                                          maxpsi=maxpsi,
                                          hwp_div=hwp_div,
                                          dead_time=dead_time
                                         )

    band = 150
    acq_qubic = QubicAcquisition(band, pointings,
                                 kind='IQU',
                                 nside=nside,
                                 detector_nep=detector_nep,
                                 detector_fknee=fknee
                                )
    ndet = 992 if not debug_mode else 2
    acq_qubic = acq_qubic[:ndet]
    ## acq_qubic.comm = acq_qubic.comm.Dup()
    ## acq_qubic.sampling.__dict__['comm'] = acq_qubic.sampling.comm.Dup()
    ## acq_qubic.scene.__dict__['comm'] = acq_qubic.scene.comm.Dup()
    ## acq_qubic.instrument.detector.__dict__['comm'] = acq_qubic.instrument.detector.comm.Dup()
    coverage = acq_qubic.get_coverage()
    if rank == 0:
        file_name = 'coverage_angspeed{}_delta_az{}.fits'.format(angspeed, delta_az)
        with open(directory + '/cov_map_' + file_name, 'w') as f:
            write_map(f, coverage)

    input_map = np.zeros((hp.nside2npix(nside), 3))
    acq_planck = PlanckAcquisition(band, acq_qubic.scene, true_sky=input_map)
    acq_fusion = QubicPlanckAcquisition(acq_qubic, acq_planck)

    obs_noiseless = acq_fusion.get_observation(noiseless=True)
    H = acq_fusion.get_operator()
    invntt = acq_fusion.get_invntt_operator()
    A = H.T * invntt * H

    for realization in xrange(nrealizations):
        print 'rank={}: realization {} / {}'.format(rank, realization + 1, nrealizations)

        obs = obs_noiseless + acq_fusion.get_noise()
        b = H.T * invntt * obs

        solution = pcg(A, b, disp=True, maxiter=maxiter)
        rec_map = solution['x']

        if rank == 0:
            file_name = 'angspeed{}_delta_az{}_realization{}.fits'.format(angspeed,
                                                                          delta_az,
                                                                          realization)
            with open(directory + '/rec_map_' + file_name, 'w') as f:
                write_map(f, rec_map, mask=coverage>0.)

    # release the pointing matrix
    del H, A
    gc.collect()
    print rank, 'got here'

print('finished')
