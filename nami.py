# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 12:12:50 2016

@author: tharshi sri, tsrikann@physics.utoronto.ca

nami.py: A feed forward neural network with backpropagation and multiple methods
of minimization. This network was designed for use in regression problems. The
program is not optimized for GPU usage. Classification is currently in progress.

070616:
    -added documentation and initialization of weights (weights + bias)
    -ability to plot weights
    -added forward prop function
070916:
    -added helper functions get_params and set_params for wrapping and 
    unwrapping parameter vector
071016:
    -added regularization to cost function
071816:
    -added numerical gradient unit testing
    -added 2d nonlinear example
071916:
    -added 1d nonlinear fitting example
072216:
    -removed regularization of biases to let saturation of neurons occur
080116:
    -added genetic algorithm option via "method" argument: 
    train(self, x_train, y_train, x_test, y_test, method='BFGS')
081016:
    -added function to plot weights 'plot_weights()'
081216:
    -added support for multiple activations
    -added smart initialization of weights (depending on number of inputs)
081316:
    -fixed examples of classificaion and regression to be cleaner
"""

# Import 3rd party packages
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as plt
from init_plotting import init_plotting

class Network(object):
    
    def __init__(self, layers, N, reg=0, io=False, problem='regression', activation='tanh'):
        """Initializes the topology of the network and sets up random weights
        for each layer. Note that h_layers does not have to include the input
        and output layer.
        
        layers : A list of integers where each int represents the number of 
        non-bias neurons in the layer.
        
        x : The input data matrix. x should be dimension(x) by (number of
        examples)
        
        y: The input data matrix. y should be dimension(y) by (number of
        examples)
        
        reg: The amount of L2 regularization to use on the cost function.
        
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.N = N
        self.reg = reg
        self.problem = problem
        self.activation = activation
        # initialize weights and biases
        self.sizes = list(zip(self.layers[:-1], self.layers[1:]))
        self.weights = [np.random.normal(scale = np.sqrt(1.0/l), \
                size=(s[0], s[1])) \
                for s, l in list(zip(self.sizes, self.layers[:-1]))]
        self.biases = [np.random.randn(1, t) for t in layers[1:]]
        self.weight_shapes = [w.shape for w in self.weights]
        self.bias_shapes = [b.shape for b in self.biases]
        self.num_weights = sum([int(r*s) for r, s in self.weight_shapes])
        self.num_biases = sum([int(t*u) for t, u in self.bias_shapes])
        self.num_params = self.num_weights + self.num_biases
        # initialize derivative matrices
        self.dws = [np.zeros((r, s)) for r, s in self.sizes]
        self.dbs = [np.zeros((1, t)) for t in layers[1:]]
        
        if io:
            print('################################################')
            print('Feed Forward Neural Network for Regression')
            print('################################################')
            print('number of layers: {}'.format(self.num_layers))
            print('number of training examples: {}'.format(self.N))
            print('regularization parameter: {}'.format(self.reg))
            print('weight shapes: {}'.format(self.weight_shapes))
            print('bias shapes: {}'.format(self.bias_shapes))
            print('number of parameters (Ws + Bs): {0} + {1} = {2}'\
                .format(self.num_weights, self.num_biases, self.num_params))
            print('------------------------------------------------\n')
    
    def info(self):
        print('number of layers: {}'.format(self.num_layers))
        print('number of training examples: {}'.format(self.N))
        print('regularization parameter: {}'.format(self.reg))
        print('weight shapes: {}'.format(self.weight_shapes))
        print('bias shapes: {}'.format(self.bias_shapes))
        print('number of parameters (Ws + Bs): {0} + {1} = {2}'\
            .format(self.num_weights, self.num_biases, self.num_params))
        print('------------------------------------------------\n')
        
    def plot_weights(self, plot_type='image', init_label=0):
        w_idx = 2
        prefix = '_'
        if init_label:
            prefix = 'init_'
        if plot_type == 'image':
            for w in self.weights:
                fw = plt.figure()
                im = plt.imshow(w, interpolation='none', cmap='viridis')
                cbar = plt.colorbar(im, pad=0.05)
                title = 'W{}_{}'.format(w_idx, self.layers, self.reg)
                plt.title(title)
                savename  = prefix + title + '_weight_images'
                plt.savefig(savename, extension='png', dpi=300)
                w_idx += 1
        if plot_type == 'histogram':
            for w in self.weights:
                fw = plt.figure()
                im = plt.hist(w.flatten(), alpha=0.5, bins=100)
                title = prefix + 'W{}_{}'.format(w_idx, self.layers, self.reg)
                plt.title(title)
                savename  = title + '_weight_hists'
                plt.savefig(savename, extension='png', dpi=300)
                w_idx += 1
    
    def g(self, z):
        """Return nonlinear transformation to neuron output."""
        if self.activation == 'tanh':
            return 1.7159*np.tanh(2*z/3)
        if self.activation == 'relu':
            return np.log(1 + np.exp(z))
        if self.activation == 'sigmoid':
            return 1.0/(1 + np.exp(-z))
        
    def g_prime(self, z):
        """Return derivative of nonlinear activation function."""
        if self.activation == 'tanh':
            return 1.14393*(1 - np.tanh(2*z/3)**2)
        if self.activation == 'relu':
            return 1.0/(1 + np.exp(-z))
        if self.activation == 'sigmoid':
            return self.g(z)*(1 - self.g(z))
        
    def h(self, z):
        """Return the output transformation."""
        return z
            
    def h_prime(self, z):
        """Return the derivative of the output transformation."""
        return 1

    def forward(self, x, neuron_info=0):
        """Return the final activation of the input matrix x. If neuron_info is
        on then a list on neuron inputs and outputs are returned in a list per
        layer.
        
        """
        zs = []
        acts = [x]
        a = x
        for w, b in list(zip(self.weights[:-1], self.biases[:-1])):
            # notice that + b underneath is broadcast row-wise
            z = np.dot(a, w) + b
            zs.append(z)
            a = self.g(z)
            acts.append(a)
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        zs.append(z)
        a = self.h(z)
        acts.append(a)
        if neuron_info:
            ans = a, zs, acts[:-1]
        else:
            ans = a
        return ans
        
    def cost(self, x, y):
        """Return the squared cost with optional L2 regularization for 
        data {x, y}.
        
        """
        yhat = self.forward(x)
        diff = yhat - y
        ps = self.get_params()
        J_0 = 0.5*np.linalg.norm(diff)**2/self.N
        J_r = 0.5*self.reg*np.linalg.norm(ps[:self.num_weights])**2/self.num_weights
        ratio = J_0/J_r
        print(ratio)
        return (J_0 + J_r)
    
    def cost_prime(self, x, y):
        """return the lists of derivatives dNet/dW and dNet/dB."""
        yhat, zs, acts = self.forward(x, neuron_info=1)
        cnst = 1.0/self.N
        diff = yhat - y
        delta = cnst*diff*self.h_prime(zs[-1])
        self.dws[-1] = (np.dot(acts[-1].T, delta) \
                + (self.reg/self.num_weights)*self.weights[-1])
        self.dbs[-1] = np.sum(delta, axis=0)
        w_previous = self.weights[-1]
        # loop backwards and calculate the rest of the derivatives !!!
        for i in range(self.num_layers - 2, 0, -1):
            w_current = self.weights[i - 1]
            w_previous = self.weights[i]
            z = zs[i - 1]
            a = acts[i - 1]
            delta = np.dot(delta, w_previous.T)*self.g_prime(z)
            self.dws[i - 1] = np.dot(a.T, delta) \
                + (self.reg/self.num_weights)*w_current
            self.dbs[i - 1] = np.sum(delta, axis=0)
            w_previous = w_current
        return self.dws, self.dbs
        
    def set_params(self, params, retall=0):
        """Change single parameter array into weight/bias matricies."""
        idx = 0
        start = 0
        for w in self.weights:
            temp = self.sizes[idx][0]
            tempp = self.sizes[idx][1]
            end = start + temp*tempp
            self.weights[idx] = params[start:end].reshape((temp, tempp))
            start = end
            idx += 1
        idx = 0
        for b in self.biases:
            temppp = self.layers[1:][idx]
            end = start + temppp
            self.biases[idx] = params[start:end].reshape((1, temppp))
            start = end
            idx += 1
        if retall:
            return self.weights, self.biases
   
    def get_params(self):
        """Get all weights/biases rolled into one array."""
        params = self.weights[0].ravel()
        for w in self.weights[1:]:
            params = np.concatenate((params, w.ravel()))
        for b in self.biases:
            params = np.concatenate((params, b.ravel()))
        return params
     
    def get_grads(self, x, y):
        """Get all weights/biases rolled into one array."""
        dws, dbs = self.cost_prime(x, y)
        grads_list = [dw.ravel() for dw in dws] + [db.ravel() for db in dbs]
        grads = grads_list[0]
        for g in grads_list[1:]:
            grads = np.concatenate((grads, g))
        return grads
        
    def numgrad(self, x, y):
        """Get a numerical estimate of the gradient vector at (x,y)."""
        paramsInitial = self.get_params()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-5
    
        for p in range(len(paramsInitial)):
            # Set perturbation vector
            perturb[p] = e
            self.set_params(paramsInitial + perturb)
            loss2 = self.cost(x, y)
            
            self.set_params(paramsInitial - perturb)
            loss1 = self.cost(x, y)
    
            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)
    
            # Return the value we changed to zero:
            perturb[p] = 0
        
        # Return Params to original value:
        self.set_params(paramsInitial)
    
        return numgrad 

class Trainer(object):
    
    def __init__(self, Network):
        """Make Local reference to network."""
        self.net = Network
        
    def cost_wrapper(self, params, x, y):
        self.net.set_params(params)
        cost = self.net.cost(x, y)
        return cost
        
    def grad_wrapper(self, params, x, y):
        self.net.set_params(params)
        grads = self.net.get_grads(x, y)
        return grads
        
    def callback(self, params):
        self.net.set_params(params)
        self.J.append(self.net.cost(self.x, self.y))
        self.J_test.append(self.net.cost(self.x_t, self.y_t))
        
    def callbackGA(self, params, convergence=0.1):
        self.net.set_params(params)
        self.J.append(self.net.cost(self.x, self.y))
        self.J_test.append(self.net.cost(self.x_t, self.y_t))
        
    def train(self, x_train, y_train, x_test, y_test, \
        method='BFGS', \
        bounds=(0, 4)):
        # Make an internal variable for the callback function:
        self.x = x_train
        self.y = y_train
        self.x_t = x_test
        self.y_t = y_test
        self.method = method
        self.bounds = bounds[0]
        self.w_max = bounds[1]

        # Make empty list to store training/testing costs:
        self.J = []
        self.J_test = []
        
        print('Minimization using ' + self.method + ':')        
            
        if method =='GA':
            bounds = [(-self.w_max, self.w_max) for i in \
                range(len(self.net.get_params()))]
            _res = spo.differential_evolution(self.cost_wrapper, \
                bounds=bounds, args=(x_train, y_train), \
                callback=self.callbackGA, maxiter=int(2e3), \
                polish=1, disp=True)
         
        else:
            params0 = self.net.get_params()
            bounds = None
            if self.bounds:
                bounds = [(-self.w_max, self.w_max) for i in \
                range(len(self.net.get_params()))]
            _res = spo.minimize(\
                self.cost_wrapper, \
                params0, method=method, \
                jac=self.grad_wrapper, \
                args=(x_train, y_train), 
                callback=self.callback, \
                options =  \
                {'maxiter' : int(1e4), 'disp' : 1, 'maxfun' : int(1e4)}, \
                bounds = bounds)
            
        self.net.set_params(_res.x)
        self.results = _res
            
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import palettable
    init_plotting()
    width  = 7.784
    height = width / 1.618
    plt.rcParams['text.usetex'] = True
    
    print('##############################################################')
    print('Welcome to N.A.M.I.')
    print('##############################################################')
    question = input('"Classification" (0) or Regression (1) \n(default is classification)?')
    if question == '1':
        print('Regression.')
        problem = 1
    else:
        print('Classification.')
        problem = 0
    N = input('Enter the integer number of Training points to Generate\n(default 100):')
    if not N:
        N = int(1e2)
    M = input('Enter the integer number of Testing points to Generate\n(default 100):')
    if not M:
        M = int(1e2)
    print('\n ### Network Parameters ###')
    h_layers = input('Enter the number of neurons in each hidden layer' +  \
        'as a list ie, h1,h2,...,hn\n(default 10,5 for regression and 3,3,3,3 for classification):')
    if not h_layers:
        h_input = 0
        if problem:
            h_layers = [10, 5]
        if not problem:
            h_layers = [3, 3, 3]
        print(h_layers)
    else:
        h_input = 1
    act_fun = input('Pick a activation function ("relu", "tanh", or "sigmoid")\n(default is  tanh):')
    if not act_fun:
        act_fun = 'tanh'
        print(act_fun + '.')
    method = input('Enter a training method ("BFGS", "CG", "L-BFGS-B"):\n(default BFGS)')
    if not method:
        method = 'BFGS'
    reg = input('Amount of Regularization:\n(default 5e-4)')
    if not reg:
        reg = 5e-4
    reg = float(reg)
    
        
    if problem:
        init_plotting()
        cmap = palettable.colorbrewer.qualitative.Set2_8.mpl_colors
        
        x = np.random.uniform(low=-1, high=1, size=(int(N), 1))
        x_test = np.random.uniform(low=-1, high=1, size=(int(M), 1))
        def f_reg(x):
            return np.exp(-x**2)*np.sin(-np.pi*5*x)
            
        y = f_reg(x)
        y += 0.05*np.random.randn(y.shape[0], y.shape[1])
        y_test = f_reg(x_test)
        y_test += 0.05*np.random.randn(y_test.shape[0], y_test.shape[1])
        
        m = np.mean(x, axis=0)
        s = np.std(x, axis=0, ddof=1)
        
        X = (x - m)/s
        Xt = (x_test - m)/s
        Y = y
        Yt = y_test
        
        if not h_input:
            h = h_layers
        else:
            h = [int(x) for x in h_layers.split(sep=',')]
        layers = [len(X.T)] + h + [len(y.T)]
        net = Network(layers, int(N), reg=reg, activation=act_fun, io=1)
        trainer = Trainer(net)
        if method == 'L-BFGS-B':
            bounds = (1, 10)
        trainer.train(X, Y, Xt, Yt, method='BFGS')
    
        t = np.linspace(-1.01, 1.01, int(1e4)).reshape((int(1e4), 1))
        nn = net.forward((t - m)/s)
        mse = ((nn - f_reg(t))**2).mean(axis=0)
        std = ((nn - f_reg(t))**2).std(axis=0, ddof=1)
        

        title = 'Neural Network Fit with MSE = {:.3e} and Residual $\sigma$ = {:.3e}'.format(mse[0], std[0])
        
        
        init_plotting()
        ax = plt.subplot(111)
        f_reg = plt.gcf()
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('g(x)')
        ax.scatter(x_test, y_test, s=50, marker='o', edgecolors=cmap[0],
                    facecolors='none', label='Testing Data')
        ax.plot(t, nn, color='k', label='Neural Net Estimate', linewidth = 1)
        plt.legend()
        f_reg.set_size_inches(width, height)
        savename = 'nnReg.pdf'
        plt.savefig(savename, bbox_inches='tight')        

    if not problem:    
    
        x = np.random.uniform(low=0, high=10, size=(int(N), 2))
        x_test = np.random.uniform(low=0, high=10, size=(int(M), 2))
        y = np.zeros((int(M), 1))
        y_test = np.zeros((int(M), 1))
        
        def f_c(x1, x2):
            """test function."""
            if (x1 - 5)**2 + (x2 - 4)**2 <= 3**2:
                return 1
            else:
                return 0
                
        for i in range(len(y)):
            y[i] = f_c(x[i, 0], x[i, 1])
            
        for i in range(len(y_test)):
            y_test[i] = f_c(x[i, 0], x[i, 1])
            
        m = np.mean(x, axis=0)
        s = np.std(x, axis=0, ddof=1)
        
        X = (x - m)/s
        Xt = (x_test - m)/s
        Y = y
        Yt = y_test
        
        if not h_input:
            h = h_layers
        else:
            h = [int(x) for x in h_layers.split(sep=',')]
        layers = [len(X.T)] + h + [len(y.T)]
        net = Network(layers, int(N), reg=reg, activation=act_fun, io=1)
        trainer = Trainer(net)
        if method == 'L-BFGS-B':
            bounds = (1, 10)
        trainer.train(X, Y, Xt, Yt, method='BFGS')
        
        a1 = np.linspace(0, 10, 100)
        a2 = np.linspace(0, 10, 100)
        a1, a2 = np.meshgrid(a1, a2)
        nn = np.zeros((a1.shape))
        
        for i in range(len(a1)):
            for j in range(len(a2)):
                temp = np.hstack(((a1[i][j] - m[0])/s[0], (a2[i][j] - m[1])/s[1]))
                nn[i][j] = net.forward(temp)
                
        mse = ((nn - y_test)**2).mean(axis=0)
        std = ((nn - y_test)**2).std(axis=0, ddof=1)
     
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        f_cl = plt.figure()
        ax = plt.subplot(111)
        ax.set_aspect(1./ax.get_data_ratio())
        title = 'Classification with MSE = {:.3e} and Residual $\sigma$ = {:.2e}'.format(mse[0], std[0])
        ax.set_title(title)
        plt.margins(0.1,0.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(x_test[:, 0], x_test[:, 1], edgecolors='k', alpha=0.6, s=20, marker='o', facecolors='None', label='Testing Data')
        plt.legend(loc='upper right')        
        CS = plt.contour(a1, a2, nn, linewidths=0.5, cmap='jet')
        circle = plt.Circle((5, 4), 3, color='k', fill=False, linewidth=1)
        ax.add_artist(circle)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(CS, cax=cax)
        cbar.lines[0].set_linewidth(10)
        
        savename = 'nnCl.pdf'
        f_cl.set_size_inches(width/1.5, width/1.5)
        f_cl.savefig(savename, bbox_inches='tight')

#%%   
    plt.clf()
    init_plotting()
    ax = plt.subplot(111)
    f_cost = plt.gcf()
    title = 'Training History, Layers = {}, Reg={:.4e}, Method = {}'.format(layers, reg, method)
    ax.set_title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(title)
    plt.loglog(trainer.J, label='Training')
    plt.loglog(trainer.J_test, label='Testing')
    plt.legend()
    savename = 'costs.pdf'
    f_cost.set_size_inches(width, height)
    f_cost.savefig(savename, bbox_inches='tight')
        
        