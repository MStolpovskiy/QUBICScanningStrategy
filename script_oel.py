from __future__ import division
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as mp

#import healpy as hp
#import numpy as np
from coverage import *

nside = 256
racenter = 0.0
deccenter = -46.0

starting_point = np.array([1, 30, 320, 1, 15])
find_optimum(starting_point)

mp.show()
