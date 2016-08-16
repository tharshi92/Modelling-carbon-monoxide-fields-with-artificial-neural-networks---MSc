# Import needed modules
import numpy as np
import matplotlib.pyplot as plt
from prepNetwork import y_max
from nami import Network, Trainer

def movingAverage(x, window):
    cumsum_vec = np.cumsum(np.insert(x, 0, 0)) 
    ma = (cumsum_vec[window:] - cumsum_vec[:-window]) / window
    return ma

data = np.load('x_red_cov.npy')
x = data[0:8760, :]
x_test = data[8760:17520, :]
y = np.load('y.npy')
y_test = np.load('yt.npy')

#%%

N = len(x)
layers = [len(x.T), 7, len(y.T)]
reg = 1e-5
net = Network(layers, N, reg, io=True)
title_prefix = str(layers)
w = 0.4
trainer = Trainer(net)
trainer.train(x, y, x_test, y_test)

#fig0 = plt.figure()
#plt.plot(trainer.J, label='Training', linewidth=w)
#plt.plot(trainer.J_test, label='Testing', linewidth=w)
#plt.xlabel('iteration')
#title = 'cost_plot_' + title_prefix
#plt.title(title)
#plt.legend()
#plt.grid('on')
#plt.savefig(title, extension='png',  dpi=300)

#%%

y_p = y_max*y
z_p = y_max*net.forward(x)
yt_p = y_max*y_test
zt_p = y_max*net.forward(x_test)

window = 24*7

yt_ma = movingAverage(yt_p, window)
zt_ma = movingAverage(zt_p, window)

t1 = np.arange(0, len(y_p))
t2 = np.arange(len(y_p), len(y_p) + len(yt_p))

t = np.arange(0, len(yt_ma))/24
t3 = np.arange(0, len(yt_p))/24

err1 = np.linalg.norm((z_p - y_p)**2)/len(y_p)
err2 = np.linalg.norm((zt_p - yt_p)**2)/len(yt_p)
err3 = np.linalg.norm((zt_ma - yt_ma)**2)/len(yt_ma)
std2 = np.std(zt_p - yt_p, ddof=1)

#%%

f = plt.figure(figsize=(14, 8))
plt.plot(t3, yt_p, alpha=0.5, label='testing data', linewidth=w)
plt.plot(t, yt_ma, label='testing data (smoothed)', linewidth=2*w)
plt.plot(t, zt_ma, 'k', label='network estimate (smoothed)', linewidth=2*w)
plt.plot(t, zt_ma - yt_ma, 'r', label='difference', linewidth=2*w)

plt.xlabel('Days Since Jan 1st 2007')
ylabel = 'Mean CO Field (ppbv), Smoothing Window = {} hrs'.format(window)
plt.ylabel(ylabel)
title1 = 'Results (Reduced): Mean SSE = {:e}, Standard Deviation = {:e}'
title2 = ', Layers: {}, regularization coeff = {}'
title = title1 + title2
plt.title(title.format(err2, std2, layers, reg))
plt.legend(fontsize=10)
plt.grid('on')
savename = 'nnfieldred_{}_{}_{}'.format(window, reg, layers)
plt.savefig(savename, extension='png', dpi=300)
