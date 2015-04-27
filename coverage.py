#import matplotlib
#matplotlib.use('Agg')

#import matplotlib.pyplot as mp
import numpy as np
import healpy as hp
from myqubic import (create_sweeping_pointings, mask_pointing)
from qubic import QubicAcquisition
from pyoperators.memory import ones
from copy import copy

def oel(point, nside=256, verbose=False, ndet_for_omega_and_eta=50, ndet_for_lambda=20):
    '''
    point - single point on the parameter space
    '''
    pointings = create_sweeping_pointings(angspeed=point[0],
                                          delta_az=point[1],
                                          nsweeps_per_elevation=int(point[2]),
                                          angspeed_psi=point[3],
                                          maxpsi=point[4],
                                          sampling_period=0.2)
    if verbose: print 'pointings created'

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

#    coverage, single_detector_coverages = GetCoverageAndSDCoverages(acq)
    coverage = GetCoverage(acq)
    single_detector_coverages = GetSDCoverages(acq, ndet_for_lambda)
    o = Omega(coverage)
    e = eta(coverage)
    l = overlap(single_detector_coverages)

    if verbose:
        print 'Omega = {}'.format(o)
        print 'eta = {}'.format(e)
        print 'lambda = {}'.format(l)
        
    return o, e, l

def Omega(coverage):
    nside = hp.npix2nside(len(coverage))
    return (normalized_coverage(coverage).sum() *
            hp.nside2pixarea(nside, degrees=True))

def eta(coverage):
    nside = hp.npix2nside(len(coverage))
    return (normalized_coverage(coverage).sum() / 
            (normalized_coverage(coverage)**2).sum())

def overlap(single_detector_coverages):
    nside = hp.npix2nside(len(single_detector_coverages[0]))
    ndet = len(single_detector_coverages)
    c_sum = np.zeros(len(single_detector_coverages[0]))
    c_prod = np.ones(len(single_detector_coverages[0]))
    for c in single_detector_coverages:
        c_sum += c
        c_prod *= c
    c_prod = np.power(c_prod, 1./ndet)
    c_sum /= ndet
    overlap = c_prod.sum() / c_sum.sum()
    return overlap

def normalized_coverage(coverage):
    return coverage / coverage.max()

def OneDetCoverage(acq, detnum):
    acq_ = copy(acq)
    acq_.instrument = acq_.instrument[detnum]
    H = acq_.get_operator()
    coverage = H.T(ones(H.shapeout))
    coverage[coverage < 0.] = 0.
    return coverage

def GetCoverageAndSDCoverages(acq):
    coverage = np.zeros(hp.nside2npix(acq.scene.nside))
    single_detector_coverages = []
    for detnum in xrange(len(acq.instrument)):
        odc = OneDetCoverage(acq, detnum)
        coverage += odc
        single_detector_coverages.append(odc)
    return coverage, single_detector_coverages

def GetSDCoverages(acq, ndet=None):
    single_detector_coverages = []
    if ndet == None:
        for idet in xrange(len(acq.instrument)):
            odc = OneDetCoverage(acq, idet)
            single_detector_coverages.append(odc)
        return single_detector_coverages
    for idet in (np.random.random(ndet) * len(acq.instrument)).astype(int):
        odc = OneDetCoverage(acq, idet)
        single_detector_coverages.append(odc)
    return single_detector_coverages

def GetCoverage(acq):
    H = acq.get_operator()
    coverage = H.T(ones(H.shapeout))
    coverage[coverage < 0.] = 0.
    return coverage
        
def criterium((omega, eta, overlap)):
    return omega * overlap / eta

#def criterium((omega, eta)):
#    return omega / eta

def get_new_point(current_point, previous_point, current_c, previous_c, point_boundaries, step):
    numpar = len(current_point)
    grad_p_to_c = grad(previous_point, current_point, previous_c, current_c) # gradient from previous point to the current one
    dist_p_to_c = np.sqrt(((previous_point - current_point)**2).sum()) # distance from previous point to the current one
    print '\t\tdistance of previous jump', dist_p_to_c
    # calculate gradient to the random direction
    rand_point = current_point + (np.random.random(numpar) * 2 - 1) * step # random_distance_coeff * dist_p_to_c
    rand_point[rand_point < point_boundaries[:, 0]] = point_boundaries[:, 0][rand_point < point_boundaries[:, 0]]
    rand_point[rand_point > point_boundaries[:, 1]] = point_boundaries[:, 1][rand_point > point_boundaries[:, 1]]
    print 'random_point = ', rand_point 
    rand_c = criterium(oel(rand_point)) # criterium value at the rand_point
    rand_grad = grad(rand_point, current_point, rand_c, current_c)
    
    grad_to_next = grad_p_to_c + rand_grad
    next_point = current_point + grad_to_next / np.sqrt(np.sum(grad_to_next**2)) * step
    next_point[next_point < point_boundaries[:, 0]] = point_boundaries[:, 0][next_point < point_boundaries[:, 0]]
    next_point[next_point > point_boundaries[:, 1]] = point_boundaries[:, 1][next_point > point_boundaries[:, 1]]
    next_c = criterium(oel(next_point))

    return next_point, next_c

def grad(point_1, point_2, c_1, c_2):
    grad = np.empty(len(point_1))
    for i, (p1, p2) in enumerate(zip(point_1, point_2)):
        grad[i] = (c_1 - c_2) / (p1 - p2) if p1 != p2 else 0.
    return grad

def find_optimum(starting_point, tol=1., point_boundaries=None, step=None):
    numpar = len(starting_point)
    if point_boundaries == None:
        point_boundaries = np.empty((numpar, 2))
        point_boundaries[0] = np.array([0.1, 5]) # angspeed
        point_boundaries[1] = np.array([10, 50]) # delta_az
        point_boundaries[2] = np.array([100, 500]) # nsweeps_per_elevation
        point_boundaries[3] = np.array([0, 2]) # angspeed_psi
        point_boundaries[4] = np.array([0, 20]) # maxpsi
    if step == None:
        step = (point_boundaries[:, 1] - point_boundaries[:, 0]) #  * 0.1
    previous_point = starting_point
    previous_c = criterium(oel(previous_point))
    current_point = previous_point + (np.random.random(numpar) * 2 - 1) * step
    current_point[current_point < point_boundaries[:, 0]] = point_boundaries[:, 0][current_point < point_boundaries[:, 0]]
    current_point[current_point > point_boundaries[:, 1]] = point_boundaries[:, 1][current_point > point_boundaries[:, 1]]
    current_c = criterium(oel(current_point))
    c_change = np.abs(previous_c - current_c)
    print '\tstarting point =', previous_point
    print '\tstarting criterium =', previous_c
    all_c = []
    all_points = []
#    mp.figure(1)
    while c_change > tol:
        all_c.append(previous_c)
        all_points.append(previous_point)
        next_point, next_c = get_new_point(current_point, previous_point, current_c, previous_c, point_boundaries, step)
        # if direction changes, reduce step
        step[((next_point - current_point) * (current_point - previous_point)) < 0.] /= 10
        previous_point = current_point
        current_point = next_point
        previous_c = current_c
        current_c = next_c
        c_change = np.abs(previous_c - current_c)
        print '\tpoint =', previous_point
        print '\tcriterium =', previous_c
    all_c.append(current_c)
    all_points.append(current_point)
    all_c = np.array(all_c)
    all_points = np.array(all_points)
#    mp.plot(all_points[:, 0], all_points[:, 1])
#    mp.xlim(point_boundaries[0])
#    mp.ylim(point_boundaries[1])

    print '\tfinal point =', current_point
    print '\tfinal criterium =', current_c

    # if found maximum is not a maximum among tried points, go back there and try with smaller step
    if all_c[-1] != all_c.max():
        return find_optimum(all_points[all_c == all_c.max()][0], step=step/10.)
    
    return current_point
