# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 09:11:27 2016

@author: tharshi
"""

from prepNetwork import data, data_test
import numpy as np
import matplotlib.pyplot as plt

x = np.concatenate((data, data_test))
x = x/np.array([1, 1, 1, 1, 1, 1e2, 1e6]) # pbl in decametre and source is kiloton
mu = np.mean(x, axis = 0)
sigma = np.std(x, axis = 0, ddof=1)
n = len(x)

z = (x - mu)
C = np.dot(z.T, z)/(n - 1)
C_max = np.amax(C)

#%% plot covariances

fig0 = plt.figure(figsize=(10,10))
plt.imshow(C/C_max, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Covariance Matrix, Scaled by {:e}'.format(C_max))
plt.xticks([0, 1, 2, 3, 4, 5, 6], \
['uwind', 'vwind', 'pressure', 'temperature', 'humidity', 'pblh', 'source'])
plt.yticks([0, 1, 2, 3, 4, 5, 6], \
['uwind', 'vwind', 'pressure', 'temperature', 'humidity', 'pblh', 'source'])
plt.savefig('covariance', extension='png', dpi=300)
    
#%% Use SVD to perform dimension reduction
        
U, s, V = np.linalg.svd(z, full_matrices=False)
l = s**2/(n - 1)
S = np.diag(s)
L = np.diag(l)

s_info = np.zeros(s.shape)
l_info = np.zeros(l.shape)

for i in range(len(s_info)):
    s_info[i] = sum(s[0:i + 1])/sum(s)
    l_info[i] = sum(l[0:i + 1])/sum(l)
    
# find components such that ~85% of information is captured
    
idx = np.where(l_info < 0.85)[0]
n_critical = max(idx)

pcs = np.dot(U[:, 0:n_critical + 1], S[0:n_critical + 1, 0:n_critical + 1])

#%% Plot SVD Information
# plot S and L

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.plot(s, 'b-o' , alpha=0.6, linewidth=1)
ax1.set_ylabel('singular values')

ax2.plot(l, 'g-o' , alpha=0.6, linewidth=1)
ax2.set_ylabel('eigenvalues')

plt.suptitle('spectral values from cov')
savename = 'singular_eigen_values_fromcov'
plt.savefig(savename, extension='png', dpi=300)

fig= plt.figure()
plt.plot(s_info, 'b-o' , alpha=0.6, label='singular values')
plt.plot(l_info, 'g-o' , alpha=0.6, label='eigenvalues')
plt.title('Cumulative Information from Covariance Matrix')
plt.ylabel('information')
plt.legend(loc='upper left', fontsize=10)
savename = 'cumulative_singular_eigen_values_fromcov'
plt.savefig(savename, extension='png', dpi=300)

#%% Reduced Dimension Data
    
Cr = np.dot(pcs.T, pcs)/(n - 1)
Cr_max = np.amax(Cr)

#%% Plot Reduced Components if possible

from mpl_toolkits.mplot3d import Axes3D
if n_critical <= 3:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x1 = pcs[:, 0]
    x2 = pcs[:, 1]
    x3 = pcs[:, 2]
    ax.set_title('85% of Information')
    ax.scatter(x1, x2, x3, s=3, c='r')
    ax.set_xlabel('pc1')
    ax.set_ylabel('pc2')
    ax.set_zlabel('pc3')
    plt.savefig('reduced_data_fromcov', extension='png', dpi=300)


#%% plot correlations for reduced data

fig = plt.figure(figsize=(10,10))
plt.imshow(Cr/Cr_max, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Reduced Covariance Matrix, Scaled by {:e}'.format(Cr_max))
plt.savefig('covariance_reduced', extension='png', dpi=300)

#%% save data

np.save('x_red_cov', pcs)