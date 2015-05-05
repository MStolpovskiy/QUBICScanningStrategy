from __future__ import division
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as mp

#import healpy as hp
#import numpy as np
from coverage import *
from scipy.optimize import minimize
## from scipy.optimize import fmin_tnc

nside = 256
racenter = 0.0
deccenter = -46.0

starting_point = np.array([1, 30, 320 * 0.00000001, 1, 15])
starting_point = np.array([0.6, 38.0, 304 * 0.00000001, 2., 20.])
#find_optimum(starting_point)
point_boundaries = np.empty((5, 2))
point_boundaries[0] = np.array([0.1, 3]) # angspeed
point_boundaries[1] = np.array([20, 50]) # delta_az
point_boundaries[2] = np.array([100 * 0.00000001, 500 * 0.00000001]) # nsweeps_per_elevation
#point_boundaries[3] = np.array([0.1, 2]) # angspeed_psi
#point_boundaries[4] = np.array([1, 20]) # maxpsi
#explore possibility of scanning without psi rotation later
point_boundaries[3] = np.array([0, 2]) # angspeed_psi
point_boundaries[4] = np.array([0, 20]) # maxpsi

result = minimize(criterium, starting_point, bounds=point_boundaries, tol=1e-3,
#                  method='Nelder-Mead') # pretty effective, but doesn't work with boundaries
##                  method='Powell') # doesn't work with boundaries
##                  method='CG') # doesn't work with boundaries
##                  method='BFGS') # doesn't work with boundaries
##                  method='Newton-CG') # doesn't work with boundaries
##                  method='Anneal') # doesn't work with boundaries
##                  method='dogleg') # doesn't work with boundaries
##                  method='trust-ncg') # doesn't work with boundaries
                  method='TNC') # found a wrong minimum
#                  method='L-BFGS-B') # doesn't converge, explores too small region
##                  method='COBYLA') # doesn't work with boundaries
#                  method='SLSQP') # goes out of boundaries

#result = fmin_tnc(criterium, starting_point, bounds=point_boundaries, ftol=1e-3)
