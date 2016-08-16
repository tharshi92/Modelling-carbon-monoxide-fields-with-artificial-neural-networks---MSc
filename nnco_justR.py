# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:37:04 2016

@author: tharshi
"""
# Import needed modules
import numpy as np
import matplotlib.pyplot as plt
from nami import Network, Trainer
from init_plotting import *
plt.rcParams['text.usetex'] = True
width  = 7.784
height = width / 1.618

def movingAverage(x, window):
    cumsum_vec = np.cumsum(np.insert(x, 0, 0)) 
    ma = (cumsum_vec[window:] - cumsum_vec[:-window]) / window
    return ma

x_test = np.load('xt.npy')
y_test = np.load('yt.npy')
co_mean = np.mean(y_test, axis=0)[0]
N = len(x_test)
reg = 5e-5
method = 'BFGS'
window = 24
layer_list = [\
            [8],
            [16],
            [8, 8]]
            
init_plotting()
ax = plt.subplot(111) 
fig_r = plt.gcf()
ax.set_xlabel('Days since Jan 1st 2007')
ax.set_ylabel(\
            'Residuals Smoothed by Window of {:.2f} day(s) (ppbv)'\
            .format(window/24))
ax.set_title(\
        'Effect of Structure on Estimates, Mean CO Field = {:.2f} ppbv'\
        .format(co_mean))

init_plotting()
fig_hist = plt.figure()
ax_h = plt.subplot(111)  
ax_h.set_xlabel('Residual (ppbv)')
ax_h.set_ylabel('Frequency')
ax_h.set_title(\
        'Effect of Structure on Error Distribution, Mean CO Field = {:.2f} ppbv'\
        .format(co_mean))

for layer in layer_list:
    nn_structure = [l for l in layer]
    nn_structure.insert(0, len(x_test.T))
    nn_structure.append(len(y_test.T))
    
    params = np.load('weights_{}.npy'.format(nn_structure))
    net = Network(nn_structure, N, reg, io=False)
    net.set_params(params)

    r = (net.forward(x_test) - y_test)
    r_ma = movingAverage(r, window)
    
    t = np.arange(0, len(r))/24
    t_ma = np.arange(0, len(r_ma))/24
    

    err = np.linalg.norm(r**2)/len(r)
    err_ma = np.linalg.norm(r_ma**2)/len(r_ma)
    std = np.std(r, ddof=1)
    label = '{}, MSE = {:.2f}, $\sigma$ = {:.2f}'.format(nn_structure, err, std)
    ax.plot(t_ma, r_ma, label=label)
    ax.legend()
    
    ax_h.hist(r, label=label, histtype='step', bins=20)
    ax_h.legend()

fig_r.set_size_inches(width, height)
fig_hist.set_size_inches(width, height)
fig_r.savefig('residuals_window_{}.pdf'.format(window), bbox_inches='tight')
fig_hist.savefig('histograms.pdf', bbox_inches='tight')
