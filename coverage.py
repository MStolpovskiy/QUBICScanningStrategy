import numpy as np
import healpy as hp
from myqubic import (create_sweeping_pointings, mask_pointing)
from qubic import QubicAcquisition
from pyoperators.memory import ones
from copy import copy

def oel(point, nside=256, verbose=False):
    '''
    point - single point on the parameter space
    '''
    pointings = create_sweeping_pointings(angspeed=point[0],
                                          delta_az=point[1],
                                          nsweeps_per_elevation=int(point[2]),
                                          angspeed_psi=point[3],
                                          maxpsi=point[4])
    if verbose: print 'pointings created'

    band = 150
    acq = QubicAcquisition(band, pointings,
                           kind='I',
                           nside=nside)
    acq.instrument = acq.instrument[:len(acq.instrument)/2]

    if verbose:
        print '-----------------------------------------------------------------------'
        print '| angspeed = {}'.format(point[0])
        print '| delta_az = {}'.format(point[1])
        print '| nsweeps_per_elevation = {}'.format(point[2])
        print '| angspeed_psi = {}'.format(point[3])
        print '| maxpsi = {}'.format(point[4])
        print '-----------------------------------------------------------------------'

    coverage, single_detector_coverages = GetCoverage(acq)
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

def GetCoverage(acq):
    coverage = np.zeros(hp.nside2npix(acq.scene.nside))
    single_detector_coverages = []
    for detnum in xrange(len(acq.instrument)):
        odc = OneDetCoverage(acq, detnum)
        coverage += odc
        single_detector_coverages.append(odc)
    return coverage, single_detector_coverages
        
def criterium((omega, eta, overlap)):
    return omega * overlap / eta

def get_new_point(point, div, c, point_ranges):
    numpar = len(point)
    new_point = np.empty(numpar)
    for i in xrange(numpar):
        new_point[i] = point[i] + c / div[i]
        if new_point[i] < point_ranges[i][0]:
            new_point[i] = point_ranges[i][0]
        if new_point[i] > point_ranges[i][1]:
            new_point[i] = point_ranges[i][1]
    (omega, eta, overlap) = oel(new_point)
    c_new = criterium((omega, eta, overlap))
    div_new = np.empty(numpar)
    for i in xrange(numpar):
        if new_point[i] == point[i]:
            div_new[i] = 0.
        else:
            div_new[i] = (c_new - c) / (new_point[i] - point[i])
    return new_point, div_new, c_new

def find_optimum(starting_point, tol=1., point_ranges=None):
    numpar = len(starting_point)
    if point_ranges == None:
        point_ranges = np.empty((numpar, 2))
        point_ranges[0] = np.array([0.1, 5]) # angspeed
        point_ranges[1] = np.array([10, 50]) # delta_az
        point_ranges[2] = np.array([100, 500]) # nsweeps_per_elevation
        point_ranges[3] = np.array([0, 2]) # angspeed_psi
        point_ranges[4] = np.array([0, 20]) # maxpsi
    div = [1e-5, 1e-2, 1., 1e-5, 1e-2]
    point = starting_point
    c = criterium(oel(point))
    c_change = 10.
    print 'starting point =', point
    print 'starting div =', div
    print 'starting criterium =', c
    while c_change > tol:
        c_old = c
        point, div, c = get_new_point(point, div, c, point_ranges)
        c_change = np.abs(c - c_old)
        print 'point = ', point
        print 'div = ', div
        print 'criterium = ', c
    return point
