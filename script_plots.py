import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as mp
import numpy as np
import healpy as hp
from cPickle import load
from qubic import QubicAcquisition, read_spectra
from myqubic import QubicAnalysis, create_sweeping_pointings
from glob import glob
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')

p_name = 'angspeed'
p_vals = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

da = []
o = np.empty(len(p_vals))
e = np.empty(len(p_vals))
l = np.empty(len(p_vals))

for p in p_vals:
    print p_name, '=', p
    pattern = './scan_ss_' + p_name + str(p) + 'seed*.pkl'
    for i, file_name in enumerate(glob(pattern)):
        print file_name
        with open(file_name, 'r') as f:
            d = load(f)
        if i == 0:
            da.append(d)
        else:
            da[-1]['ItoQU'] = np.concatenate((da[-1]['ItoQU'], d['ItoQU']), axis=0)
            da[-1]['Noise'] = np.concatenate((da[-1]['Noise'], d['Noise']), axis=0)
            da[-1]['QUmix'] = np.concatenate((da[-1]['QUmix'], d['QUmix']), axis=0)
        if 'Omega' in d.keys():
            print 'Omega =', d['Omega']
            o[p_vals == p][0] = d['Omega']
        if 'Eta' in d.keys():
            print 'eta =', d['Eta']
            e[p_vals == p][0] = d['Eta']

ItoQU_v = np.empty((len(da), 4, 3))
ItoQU_e = np.empty((len(da), 4, 3))
Noise_v = np.empty((len(da), 4, 3))
Noise_e = np.empty((len(da), 4, 3))
QUmix_v = np.empty((len(da), 4, 3))
QUmix_e = np.empty((len(da), 4, 3))

nrealizations = np.empty(len(p_vals))
for i, d in enumerate(da):
    print p_name, '=', d[p_name], ':', len(d['ItoQU']), 'realizations'
    nrealizations[i] = len(d['ItoQU'])
    ItoQU_v[i] = np.average(d['ItoQU'], axis=0)
    ItoQU_e[i] = np.std(d['ItoQU'], axis=0)
    Noise_v[i] = np.average(d['Noise'], axis=0)
    Noise_e[i] = np.std(d['Noise'], axis=0)
    QUmix_v[i] = np.average(d['QUmix'], axis=0)
    QUmix_e[i] = np.std(d['QUmix'], axis=0)

mp.figure()
mp.plot(nrealizations)
mp.savefig('scan_' + p_name + '_nreal.png')

def Plot(vals, errs, x_vals_, title):
    x_vals = x_vals_.copy()
    mp.figure()
    plot = np.empty(len(vals))
    yerr = np.empty(len(vals))
    for IQU, tIQU in enumerate('IQU'): # loop over Stocks parameters
        if IQU == 0: ax = mp.subplot(3, 1, IQU+1)
        else: ax = mp.subplot(3, 1, IQU+1, sharex=ax)
        for cov_range, (color, cr) in enumerate(zip('rygb', ['0.2-0.4','0.4-0.6','0.6-0.8','0.8-1.0'])): # loop over coverage ranges
            for i, (v, e) in enumerate(zip(vals, errs)): # loop over x_values
                plot[i] = v[cov_range, IQU]
                yerr[i] = e[cov_range, IQU]
            mp.errorbar(x_vals, plot, yerr=yerr, color=color, label='coverage in range ' + cr)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            l = ax.legend(bbox_to_anchor=(1., .5), loc='center left')
            tit = {'ItoQU': 'I to QU leakage; ',
                   'Noise': 'Noise influence; ',
                   'QUmix': 'QU mixing; '}[title]
            l.set_title(tit + tIQU, prop=fontP)
    mp.savefig('scan_' + p_name + '_' + title + '.png')

for v, e, t in zip((ItoQU_v, Noise_v, QUmix_v), (ItoQU_e, Noise_e, QUmix_e), ('ItoQU', 'Noise', 'QUmix')):
    print p_vals
    Plot(v, e, p_vals, t)

for p in p_vals:
    spectra = read_spectra(0)
    input_map = np.array(hp.synfast(spectra, 256)).T
    pointings = create_sweeping_pointings(parameter_to_change=p_name, value_of_parameter=p)
    acquisition = QubicAcquisition(150, pointings,
                                   kind='IQU',
                                   nside=256)
    analysis = QubicAnalysis(acquisition, input_map, coverage_thr=0.2, pickable=False, noise=False, run_analysis=False)
    l[p_vals == p][0] = analysis.Lambda(ndet=10)

mp.figure()
ax = mp.subplot(3, 1, 1, title='$\Omega$')
mp.plot(p_vals, o)
ax = mp.subplot(3, 1, 2, title='$\eta$', sharex=ax)
mp.plot(p_vals, e)
mp.subplot(3, 1, 3, title='$\lambda$', sharex=ax)
mp.plot(p_vals, l)
mp.savefig('scan_' + p_name + '_oel.png')
