import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
import healpy as hp
import numpy as np
from coverage import *
from scipy.optimize import minimize
import pyfits as pf
from glob import glob
from matplotlib import cm
import os
from pyoperators import MPI

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size
    
nside = 256
racenter = 0.0
deccenter = -46.0

scale = 1e-7

angspeeds = np.arange(0., scale, 0.1/2.9*scale) # np.arange(0.1, 3., 0.1)
delta_azs = np.arange(0., scale, 1./30.*scale) # np.arange(20, 50, 1)
criteria = np.empty((len(angspeeds), len(delta_azs)))

file_name = 'criteria_on_angspeed_delta_az_plane_debug_mpi{}.fits'.format(size)
if len(glob(file_name)) == 0:
    for i, angspeed in enumerate(angspeeds):
        for j, delta_az in enumerate(delta_azs):
            point = np.array([angspeed, delta_az])
            criteria[i, j] = criterium(point)
    hdu = pf.PrimaryHDU(criteria)
    if rank == 0: hdu.writeto(file_name)
else:
    hdulist = pf.open(file_name)
    criteria = hdulist[0].data
angspeeds = angspeeds / scale * 2.9 + 0.1
delta_azs = delta_azs / scale * 30. + 20.
    
point_boundaries = np.empty((2, 2))
point_boundaries[:, 0] = 0.
point_boundaries[:, 1] = scale

methods = ['Nelder-Mead', # 0 found bad minimum
           'Powell', # 1 doesn't work
           'CG', # 2 doesn't work
           'BFGS', # 3 doesn't work
           'Newton-CG', # 4 Jacobian is required
           'Anneal', # 5 doesn't work
           'dogleg', # 6 Jacobian is required
           'trust-ncg', # 7 Jacobian is required
           'TNC', # 8 explores too small area
           'L-BFGS-B', # 9 found a wrong minimum
           'COBYLA', # 10 went out of boundaries | just stucked
           'SLSQP', # 11 found a wrong minimum
           'my' # 12
           ]
method = methods[8]
file_name = 'minimization_steps_{}_mpi{}.log'.format(method, size)
if len(glob(file_name)) == 0:
    if len(glob('oel.log')) != 0:
        if rank == 0:
            os.remove('oel.log')
    point = np.array([0.5, 0.5]) * scale
    if method != 'my':
        result = minimize(criterium, point, bounds=point_boundaries, tol=1e-3,
                          method=method)
    else:
        find_optimum(point)
    if rank == 0: os.rename('oel.log', file_name)
if rank == 0:
    with open(file_name, 'r') as f:
        lines = f.readlines()
    a = np.array(lines[::2]).astype(float)
    d = np.array(lines[1::2]).astype(float)

    # ploting
    fig = mp.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot_trisurf(angspeeds, delta_azs, criteria, cmap=cm.jet, linewidth=0.2)
    X, Y = np.meshgrid(angspeeds, delta_azs)
    #ax.plot_surface(X, Y, criteria.T, rstride=1, cstride=1, cmap=cm.jet, #coolwarm,
    #        linewidth=0, antialiased=False)
    mp.contour(X, Y, np.log(criteria.T))
    #Axes3D
    mp.plot(a, d, color='r')
    mp.plot(a[0], d[0], marker='o', color='r')
    mp.plot(a[-1], d[-1], marker='o', color='b')

    mp.show()
