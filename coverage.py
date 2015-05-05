#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as mp
import numpy as np
import healpy as hp
from myqubic import (create_sweeping_pointings, mask_pointing)
from qubic import QubicAcquisition
from pyoperators.memory import ones
from copy import copy

from pyoperators import MPI

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def oel(point,
        nside=256,
        verbose=False,
        ndet_for_omega_and_eta=50,
        ndet_for_lambda=10
        ):
    '''
    point - single point on the parameter space
    '''
    if ndet_for_lambda > ndet_for_omega_and_eta:
        ndet_for_lambda = ndet_for_omega_and_eta
    pointings = create_sweeping_pointings(angspeed=point[0],
                                          delta_az=point[1],
#                                          nsweeps_per_elevation=int(point[2]),
                                          nsweeps_per_elevation=int(point[2] / 0.00000001),
                                          angspeed_psi=point[3],
                                          maxpsi=point[4],
                                          sampling_period=0.1)
    pointings = pointings[mask_pointing(pointings)]

    band = 150
    acq = QubicAcquisition(band, pointings,
                           kind='I',
                           nside=nside)
    fullfocalplane = int(len(acq.instrument) / 2)
    alldet = np.arange(fullfocalplane)
    np.random.shuffle(alldet)
    randdet = alldet[:ndet_for_omega_and_eta]
    mask = np.zeros(fullfocalplane * 2, dtype=bool)
    for i in xrange(fullfocalplane):
        if i in randdet: mask[i] = True

    acq.instrument = acq.instrument[mask]
    
    if verbose:
        print '-----------------------------------------------------------------------'
        print '| angspeed = {}'.format(point[0])
        print '| delta_az = {}'.format(point[1])
        print '| nsweeps_per_elevation = {}'.format(point[2])
        print '| angspeed_psi = {}'.format(point[3])
        print '| maxpsi = {}'.format(point[4])
        print '-----------------------------------------------------------------------'

    coverage, single_detector_coverages = GetCoverageAndSDCoverages(acq)
#    coverage = GetCoverage(acq)
#    single_detector_coverages = GetSDCoverages(acq, ndet_for_lambda, verbose=verbose)
    if ndet_for_lambda < ndet_for_omega_and_eta:
        np.random.shuffle(single_detector_coverages)
        single_detector_coverages = single_detector_coverages[:ndet_for_lambda]
    cov_thr = 0.2
    o = Omega(coverage, cov_thr=cov_thr)
    e = eta(coverage, cov_thr=cov_thr)
    l = overlap(single_detector_coverages, cov_thr=cov_thr)

    if verbose:
        print 'Omega = {}'.format(o)
        print 'eta = {}'.format(e)
        print 'lambda = {}'.format(l)
        
    return o, e, l

def Omega(coverage, cov_thr=0.2):
    nside = hp.npix2nside(len(coverage))
    return (normalized_coverage(coverage, cov_thr=cov_thr).sum() *
            hp.nside2pixarea(nside, degrees=True))

def eta(coverage, cov_thr=0.2):
    nside = hp.npix2nside(len(coverage))
    return (normalized_coverage(coverage, cov_thr).sum() / 
            (normalized_coverage(coverage, cov_thr)**2).sum())

def overlap(single_detector_coverages, cov_thr=0.2):
    nside = hp.npix2nside(len(single_detector_coverages[0]))
    ndet = len(single_detector_coverages)
    c_sum = np.zeros(len(single_detector_coverages[0]))
    c_prod = np.ones(len(single_detector_coverages[0]))
    for c in single_detector_coverages:
        c[c < c.max() * cov_thr] = 0.
        c_sum += c
        c_prod *= c
    c_prod = np.power(c_prod, 1./ndet)
    c_sum /= ndet
    overlap = c_prod.sum() / c_sum.sum()
    return overlap

def normalized_coverage(coverage, cov_thr=0.2):
    c = coverage / coverage.max()
    c[c < 0.2] = 0.
    return c

def OneDetCoverage(acq, detnum, verbose=False):
#    if verbose: print 'detnum =', detnum
    acq_ = copy(acq)
    acq_.instrument = acq_.instrument[detnum]
    H = acq_.get_operator()
    coverage = H.T(ones(H.shapeout))
    coverage[coverage < 0.] = 0.
    return coverage

def GetCoverageAndSDCoverages(acq, verbose=False):
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    if verbose:
        print 'I am = {}/{}'.format(rank, size)
    do_parallel = False
    if size >= ndet: do_parallel = True
    coverage = np.zeros(hp.nside2npix(acq.scene.nside))
    single_detector_coverages = []
    for detnum in xrange(len(acq.instrument)):
        if do_parallel and rank == idet:
            odc = OneDetCoverage(acq, detnum, verbose=verbose)
        if not do_parallel:
            odc = OneDetCoverage(acq, detnum, verbose=verbose)
        coverage += odc
        single_detector_coverages.append(odc)
    return coverage, single_detector_coverages

def GetSDCoverages(acq, ndet=None, verbose=False):
    single_detector_coverages = []
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    if verbose:
        print 'I am = {}/{}'.format(rank, size)
    do_parallel = False
    if size >= ndet: do_parallel = True
    if ndet == None:
        for idet in xrange(len(acq.instrument)):
            if do_parallel and rank == idet:
                odc = OneDetCoverage(acq, idet, verbose=verbose)
            if not do_parallel:
                odc = OneDetCoverage(acq, idet, verbose=verbose)
            single_detector_coverages.append(odc)
        return single_detector_coverages
    all_det = np.arange(len(acq.instrument))
    np.random.shuffle(all_det)
    for idet in all_det[:ndet]:
        if do_parallel and rank == idet:
            odc = OneDetCoverage(acq, idet, verbose=verbose)
        if not do_parallel:
            odc = OneDetCoverage(acq, idet, verbose=verbose)
        single_detector_coverages.append(odc)
    return single_detector_coverages

def GetCoverage(acq):
    H = acq.get_operator()
    coverage = H.T(ones(H.shapeout))
    coverage[coverage < 0.] = 0.
    return coverage
        
#def criterium((omega, eta, overlap)):
#    return omega * overlap / eta

def criterium(point):
    o, e, l = oel(point, verbose=True)
#    o, e = oel(point, verbose=True, ndet_for_omega_and_eta=1, ndet_for_lambda=1)
    c = e / o / l
    print color.BOLD + 'criterium =', c, color.END
    return c

#def criterium((omega, eta)):
#    return omega / eta

def get_new_point(current_point, previous_point, current_c, previous_c, point_boundaries, step):
    numpar = len(current_point)
    grad_p_to_c = grad(previous_point, current_point, previous_c, current_c) # gradient from previous point to the current one
    dist_p_to_c = np.sqrt(((previous_point - current_point)**2).sum()) # distance from previous point to the current one
    print '\t\tdistance of previous jump', dist_p_to_c
    # calculate gradient to the random direction
    rand_point = current_point + (np.random.random(numpar) * 2 - 1) * step / 10. # random_distance_coeff * dist_p_to_c
    rand_point[rand_point < point_boundaries[:, 0]] = point_boundaries[:, 0][rand_point < point_boundaries[:, 0]]
    rand_point[rand_point > point_boundaries[:, 1]] = point_boundaries[:, 1][rand_point > point_boundaries[:, 1]]
    print 'random_point =', rand_point 
    rand_c = criterium(oel(rand_point)) # criterium value at the rand_point
    print 'random criterium =', rand_c
    rand_grad = grad(rand_point, current_point, rand_c, current_c)
    
    grad_to_next = grad_p_to_c + rand_grad
    next_point = current_point + grad_to_next / np.sqrt(np.sum(grad_to_next**2)) * step
    next_point[next_point < point_boundaries[:, 0]] = point_boundaries[:, 0][next_point < point_boundaries[:, 0]]
    next_point[next_point > point_boundaries[:, 1]] = point_boundaries[:, 1][next_point > point_boundaries[:, 1]]
    next_c = criterium(oel(next_point))

    return next_point, next_c, rand_point, rand_c

def grad(point_1, point_2, c_1, c_2):
    grad = np.empty(len(point_1))
    for i, (p1, p2) in enumerate(zip(point_1, point_2)):
        grad[i] = (c_1 - c_2) / (p1 - p2) if p1 != p2 else 0.
    return grad

def find_optimum(starting_point, tol=5., point_boundaries=None, step=None):
    numpar = len(starting_point)
    if point_boundaries == None:
        point_boundaries = np.empty((numpar, 2))
        point_boundaries[0] = np.array([0.1, 5]) # angspeed
        point_boundaries[1] = np.array([10, 50]) # delta_az
        point_boundaries[2] = np.array([100, 500]) # nsweeps_per_elevation
        point_boundaries[3] = np.array([0, 2]) # angspeed_psi
        point_boundaries[4] = np.array([0, 20]) # maxpsi
    if step == None:
        prim_step = np.array((point_boundaries[:, 1] - point_boundaries[:, 0]) * 0.3)
        step = copy(prim_step)
    else:
        prim_step = copy(step)
    previous_point = starting_point
    previous_c = criterium(oel(previous_point))
    current_point = previous_point + (np.random.random(numpar) * 2 - 1) * step
    current_point[current_point < point_boundaries[:, 0]] = point_boundaries[:, 0][current_point < point_boundaries[:, 0]]
    current_point[current_point > point_boundaries[:, 1]] = point_boundaries[:, 1][current_point > point_boundaries[:, 1]]
    current_c = criterium(oel(current_point))
    c_change = np.abs(previous_c - current_c)
    print '\tstarting point =', previous_point
    print '\tstarting criterium =', previous_c
#    mp.figure(1)
    all_c = np.array([previous_c, previous_c])
    all_points = np.array([previous_point, previous_point])
    while c_change > tol:
        next_point, next_c, rand_point, rand_c = get_new_point(current_point, previous_point, current_c, previous_c, point_boundaries, step)
        all_c = np.append(all_c, rand_c)
        all_points = np.append(all_points, [rand_point], axis=0)
        # if direction changes, reduce step
        step[((next_point - current_point) * (current_point - previous_point)) < 0.] /= 2.
        previous_point = current_point
        current_point = next_point
        previous_c = current_c
        current_c = next_c
        all_c = np.append(all_c, previous_c)
        all_points = np.append(all_points, [previous_point], axis=0)
        c_change = np.abs(previous_c - current_c)
        print '\tpoint =', previous_point
        print '\tcriterium =', previous_c
        dist_curr_to_max = np.sqrt(((current_point - all_points[all_c == all_c.max()][0])**2).sum())
        if dist_curr_to_max > np.sqrt((step**2).sum()):
            print 'too far from maximum'
            break
    all_c = np.append(all_c, current_c)
    all_points = np.append(all_points, [current_point], axis=0)
#    mp.plot(all_points[:, 0], all_points[:, 1])
#    mp.xlim(point_boundaries[0])
#    mp.ylim(point_boundaries[1])

    print '\tfinal point =', current_point
    print '\tfinal criterium =', current_c
    
    # if found maximum is not a maximum among tried points, go back there and try with smaller step
    if all_c[-1] != all_c.max():
        return find_optimum(all_points[all_c == all_c.max()][0], step=prim_step/2.)
    
    return current_point
