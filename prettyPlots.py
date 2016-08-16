# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 01:38:21 2016

@author: tharshi
"""

import matplotlib.pyplot as plt
import numpy as np
from init_plotting import init_plotting

width  = 7.784
height = width / 1.618

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

x = np.linspace(-7, 7, 1000)
step = np.hstack((-1*np.ones((1, 500)), np.ones((1, 500)))).flatten()

plt.rcParams['text.usetex'] = True
init_plotting()
ax = plt.subplot(111)
fig = plt.gcf()
ax.set_ylabel('Activation')
ax.set_xlabel('Input Signal')
ax.set_title('Common Activation Functions')
ax.margins(0.0, 0.1)
ax.plot(x, 2*sigmoid(x) - 1, label='Sigmoid')
ax.plot(x, np.tanh(x), label='Tanh')
ax.plot(x, step, label='Step')
plt.legend(loc='lower right')
fig.set_size_inches(width, height)
plt.savefig('activation_functions.pdf', bbox_inches='tight')