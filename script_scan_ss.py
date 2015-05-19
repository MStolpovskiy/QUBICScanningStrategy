from __future__ import division

from myqubic import (create_sweeping_pointings,
                     mask_pointing)
from qubic import (QubicAcquisition,
                   PlanckAcquisition,
                   QubicPlanckAcquisition)
from qubic.io import write_map
import healpy as hp
import numpy as np
from pyoperators import MPI, pcg
import ConfigParser
import json
from itertools import product
import os

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

if rank == 0:
    path = './'
    directory = path + 'angspeed-delta_az_scan/'
    directory += 'const_el_time{}_angspeed_psi{}_dead_time{}_fknee{}_nside{}'.format(time_on_const_elevation,
                                                                                     angspeed_psi,
                                                                                     dead_time,
                                                                                     fknee,
                                                                                     nside)
    if not os.path.exists(directory):
        os.makedirs(directory)

for (angspeed, delta_az) in product(angspeed_ar, delta_az_ar):
    if rank == 0: print 'Angspeed, delta_az =', angspeed, delta_az
    coverage = np.zeros(hp.nside2npix(nside))
    for realization in xrange(nrealizations):
        if rank == 0: print 'realization', realization
        nsw = int(time_on_const_elevation / (delta_az / angspeed))
        pointings = create_sweeping_pointings(sampling_period=sampling_period,
                                              angspeed=angspeed,
                                              delta_az=delta_az,
                                              nsweeps_per_elevation=nsw,
                                              angspeed_psi=angspeed_psi,
                                              maxpsi=maxpsi,
                                              hwp_div=hwp_div
                                             )
        nep_normalization = np.sqrt(nep_normalization / (len(pointings)*pointings.period))
        pointings = pointings[mask_pointing(pointings, dead_time=dead_time)]
        
        band = 150
        detector_nep = nep / nep_normalization
        acq_qubic = QubicAcquisition(band, pointings,
                                     kind='IQU',
                                     nside=nside,
                                     detector_nep=detector_nep
                                    )
        acq_qubic = acq_qubic[:992]
        if realization == 0:
            coverage = acq_qubic.get_coverage()

        input_map = np.zeros((hp.nside2npix(nside), 3))
        acq_planck = PlanckAcquisition(band, acq_qubic.scene, true_sky=input_map)
        acq_fusion = QubicPlanckAcquisition(acq_qubic, acq_planck)

        obs = acq_fusion.get_observation()
        H = acq_fusion.get_operator()
        invntt = acq_fusion.get_invntt_operator()

        A = H.T * invntt * H
        b = H.T * invntt * obs

        solution = pcg(A, b, disp=True)

        rec_map = solution['x']

        if rank == 0:
            file_name = 'angspeed{}_delta_az{}_realization{}.fits'.format(angspeed,
                                                                          delta_az,
                                                                          realization)
            with open(directory + '/rec_map_' + file_name, 'w') as f:
                write_map(f, rec_map, mask=coverage>0.)
            
    if rank == 0:
        file_name = 'angspeed{}_delta_az{}.fits'.format(angspeed,
                                                        delta_az,
                                                        realization)
        with open(directory + '/cov_map_' + file_name, 'w') as f:
            write_map(f, coverage)
