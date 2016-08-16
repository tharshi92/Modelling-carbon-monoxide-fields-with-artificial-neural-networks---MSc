# Coded by Tharshi Srikannathasan March 2016
# This is a 4th order Runge-Kutta Scheme to Solve the Lorentz Model

# import all needed libraries 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from init_plotting import init_plotting

# define simulation parameters
T = 16.0;    # simultion length
h = 1e-3;    # timestep
N = int(T/h);   # number of steps

params = np.array([10.0, 8.0/3.0, 28.0])
x0 = np.array([1.508870, -1.531271, 25.46091])

x = np.zeros(N);
y = np.zeros(N);
z = np.zeros(N);
dX = np.zeros(N);
dY = np.zeros(N);
dZ = np.zeros(N);

# parameters for the lorentz model
sigma = params[0];
beta = params[1];
rho = params[2];

# initial conditions
x[0] = x0[0];
y[0] = x0[1];
z[0] = x0[2];

# define dynamical equation

def dynamics(x, y, z, sigma, beta, rho):
    
    dxdt = -sigma*(x - y);
    
    dydt = x*(rho - z) - y;
    
    dzdt = x*y - beta*z;
    
    return dxdt, dydt, dzdt;

# integrate using Runge Kutta Method

for i in range(N - 1):

    p1, q1, r1 = dynamics(x[i], y[i], z[i], sigma, beta, rho);
    dX[i] = p1;
    dY[i] = q1;
    dZ[i] = r1;

    p2, q2, r2 = dynamics(x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0,\
                          sigma, beta, rho);
    p3, q3, r3 = dynamics(x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0,\
                          sigma, beta, rho);
    p4, q4, r4 = dynamics(x[i] + h*p3, y[i] + h*q3, z[i] + h*r3,\
                          sigma, beta, rho);

    x[i+1] = x[i] + h*(p1 + 2.0*p2 + 2.0*p3 + p4)/6.0;
    y[i+1] = y[i] + h*(q1 + 2.0*q2 + 2.0*q3 + q4)/6.0;
    z[i+1] = z[i] + h*(r1 + 2.0*r2 + 2.0*r3 + r4)/6.0;

# write data to file

np.save('dz', dZ)
np.save('data', np.vstack((x, y, z, dX, dY)))

#%% plot data
plt.rcParams['text.usetex'] = True
temp = np.linspace(0, len(x), endpoint=True, num=N)*h

width  = 7.784
height = width / 1.618

init_plotting()
ax = plt.subplot(111)
f = plt.gcf()
ax.set_xlabel('time (unitless)')
ax.set_title('Components of Lorentz System')
ax.margins(0.0, 0.25)
ax.plot(temp, x, label='x')
ax.plot(temp, y, label='y')
ax.plot(temp, z, label='z')
plt.legend()
f.set_size_inches(width, height)
plt.savefig('2dLorentz.pdf', bbox_inches='tight')
#%%
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.set_title('Strange Attractor in 3-Space')
#ax.plot(x, y, z)
#fig.set_size_inches(width, height)
#plt.savefig('3dLorentz.pdf', bbox_inches='tight')
