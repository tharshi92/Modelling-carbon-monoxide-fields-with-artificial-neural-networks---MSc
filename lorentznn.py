# Coded by Tharshi Srikannathasan March 2016
# This is a 4th order Runge-Kutta Scheme to Solve the Lorentz Model
# A Neural Network will be used to estimate dz/dt at every time step
# The Runge-Kutta Method will be compared to the output of the neural network assisted simulation

# import all needed libraries 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nami import Network, Trainer
from init_plotting import *


width  = 7.784
height = width / 1.618

def dynamics(x, y, z, sigma, beta, rho):
    
    dxdt = -sigma*(x - y)
    
    dydt = x*(rho - z) - y
    
    dzdt = x*y - beta*z
    
    return dxdt, dydt, dzdt

# import simulation data


data = np.load('data.npy')
data = data.T
signal = np.load('dz.npy').reshape((len(data), 1))

#%%
# train network!

mu = np.mean(data, axis=0)
s = np.std(data, axis=0, ddof=1)
X = (data - mu)/s
Y = signal

p = int(len(data)/2)
q = len(data)

x_test = X[p:q]
y_test = Y[p:q]
x_train = X[0:p]
y_train = Y[0:p]

N = int(len(x_train))

layers = [5, 100, 1]
reg = 1e-4
net = Network(layers, N, reg, io=1)
#%% train and plot cost
# train network
trainer = Trainer(net)
trainer.train(x_train, y_train, x_test, y_test, method='L-BFGS-B')
wbs = net.get_params()
np.save('wbs_lorentz', wbs)
print('done training')
#%%
plt.rcParams['text.usetex'] = True
init_plotting()
ax = plt.subplot(111)
f_c = plt.gcf()
plt.margins(0.0, 0.1)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Training History, Layers = {}, Reg={:.4e}, Method = {}'.format(layers, reg, 'BFGS'))
ax.loglog(trainer.J, label='Training')
ax.loglog(trainer.J_test, label='Testing')
plt.legend()
savename = 'costsLorentz.pdf'
f_c.set_size_inches(width, height)
plt.savefig(savename, bbox_inches='tight')
#%%
# rerun simulation with trained network

# define simulation parameters
T = 16.0    # simultion length
h = 1e-3    # timestep
N = int(T/h)   # number of steps
time = np.linspace(0, T, N)

params = np.array([10.0, 8.0/3.0, 28.0])
x0 = np.array([1.508870, -1.531271, 25.46091])

x = np.zeros(N).reshape((N, 1))
y = np.zeros(N).reshape((N, 1))
z = np.zeros(N).reshape((N, 1))
dX = np.zeros(N).reshape((N, 1))
dY = np.zeros(N).reshape((N, 1))
dZ = np.zeros(N).reshape((N, 1))
xt = np.zeros(N).reshape((N, 1))
yt = np.zeros(N).reshape((N, 1))
zt = np.zeros(N).reshape((N, 1))
dXt = np.zeros(N).reshape((N, 1))
dYt = np.zeros(N).reshape((N, 1))
dZt = np.zeros(N).reshape((N, 1))

# parameters for the lorentz model
sigma = params[0]
beta = params[1]
rho = params[2]

# initial conditions
x[0] = x0[0]
y[0]= x0[1]
z[0]= x0[2]
xt[0] = x0[0]
yt[0]= x0[1]
zt[0]= x0[2]

# integrate using Runge Kutta Method
print('Running Simulation with Neural Network')

for i in range(N - 1):

    # update nn simulation

    p1, q1, dummy = dynamics(x[i], y[i], z[i], sigma, beta, rho)
    temp1 = (np.column_stack((x[i], y[i], z[i], p1, q1)) - mu)/s
    r1 = net.forward(temp1)
                     
    dX[i] = p1
    dY[i] = q1
    dZ[i] = r1

    p2, q2, dummy = dynamics(x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0,\
                      sigma, beta, rho)
    temp2 = (np.column_stack((x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0, p2, q2)) - mu)/s
    r2 = net.forward(temp2)
                     
    p3, q3, dummy = dynamics(x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0,\
                     sigma, beta, rho)
    temp3 = (np.column_stack((x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0, p3, q3)) - mu)/s
    r3 = net.forward(temp3)
                     
    p4, q4, dummy = dynamics(x[i] + h*p3, y[i] + h*q3, z[i] + h*r3,\
                      sigma, beta, rho)
    temp4 = (np.column_stack((x[i] + h*p3, y[i] + h*q3, z[i] + h*r3, p4, q4)) - mu)/s
    r4 = net.forward(temp4)
    x[i+1] = x[i] + h*(p1 + 2.0*p2 + 2.0*p3 + p4)/6.0
    y[i+1] = y[i] + h*(q1 + 2.0*q2 + 2.0*q3 + q4)/6.0
    z[i+1] = z[i] + h*(r1 + 2.0*r2 + 2.0*r3 + r4)/6.0

    # update truth
    
    p1t, q1t, r1t = dynamics(xt[i], yt[i], zt[i], sigma, beta, rho)
    dXt[i] = p1t
    dYt[i] = q1t
    dZt[i] = r1t

    p2t, q2t, r2t = dynamics(xt[i] + h*p1t/2.0, yt[i] + h*q1t/2.0, zt[i] + h*r1t/2.0,\
                             sigma, beta, rho)
    p3t, q3t, r3t = dynamics(xt[i] + h*p2t/2.0, yt[i] + h*q2t/2.0, zt[i] + h*r2t/2.0,\
                             sigma, beta, rho)
    p4t, q4t, r4t = dynamics(xt[i] + h*p3t, yt[i] + h*q3t, zt[i] + h*r3t,\
                             sigma, beta, rho)

    xt[i+1] = xt[i] + h*(p1t + 2.0*p2t + 2.0*p3t + p4t)/6.0
    yt[i+1] = yt[i] + h*(q1t + 2.0*q2t + 2.0*q3t + q4t)/6.0
    zt[i+1] = zt[i] + h*(r1t + 2.0*r2t + 2.0*r3t + r4t)/6.0
    
err_x = np.linalg.norm(x - xt)**2/N
err_y = np.linalg.norm(y - yt)**2/N
err_z = np.linalg.norm(z - zt)**2/N

#%%
# Three subplots sharing both x/y axes
plt.rcParams['text.usetex'] = True
init_plotting()
ax = plt.subplot(111)
f = plt.gcf()
plt.margins(0.0, 0.1)
ax.set_xlabel('time (unitless)')
ax.set_ylabel('Residuals')
ax.set_title('Lorentz Component Residuals, MSE = ({:.2e}, {:.2e}, {:.2e})'.format(err_x, err_y, err_z))
ax.plot(time, x - xt, label='$\Delta$x')
ax.plot(time, y - yt, label='$\Delta$y')
ax.plot(time, z - zt, label='$\Delta$z')
plt.legend()
f.set_size_inches(width, height)
savename = 'lorentz_solution_{}_{}'.format(layers, reg) + '.pdf'
plt.savefig(savename, bbox_inches='tight')

