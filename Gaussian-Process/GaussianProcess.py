from __future__ import division
import numpy as np

import Tools
import Mean
import Covariance
import Inference

import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import solve
from numpy import diagonal

from Inference import GetSamples
# import scipy.spatial.distance as spdist


SHADEDCOLOR = [0.7539, 0.89453125, 0.62890625, 1.0]
MEANCOLOR = [ 0.2109375, 0.63385, 0.1796875, 1.0]
DATACOLOR = [0.12109375, 0.46875, 1., 1.0]



class GP(object):
    '''
        Base Class for Gaussian Process
    '''
    def __init__(self):
        super(GP, self).__init__()
        print 'Inside the init class of GP'
        note = 'Runing GP....................'

        self.meanFunc = None
        self.covFunc = None
        self.infFunc = None
        self.sampFunc = None

    def getPrior(self, x_prior=None, no_of_samples=10):
        '''
            According to Gaussian Process, The prior are the function itself. The Function is defined by the covariance function between data points. The covatiance function can be many types, such as, the squared exponential, the neural net kernel and many more

            | Prior:    P(f(x)|Mi) ~ GP(m(x)= 0, k(x,x')) Here m(x) is the zero mean vector and k(x,x') is the covariance matrix(squared exponential kernel)
                So when taken datapoints between a finite set of values (say, linspace(-5,5,50))), In such a case the prior is a distribution over function with mean = 0 and covariance k(x,x') 
        '''
        if x_prior is None:
            self.x_prior = Tools.gp_1D_X_range(-10.0, 10.01, .125)

        if not isinstance(self.x_prior, np.ndarray):
            self.x_prior = np.array(x_prior, dtype = float)

        prior_mean = self.meanFunc.getMean(a=self.x_prior)
        prior_K = self.covFunc.sq_exp_vect_scipy(a=self.x_prior, mode='self')
        prior_L = self.infFunc.cholesky(prior_K, self.x_prior.shape[0])
        self.prior_fnc = GetSamples().draw_samples(prior_L, s=no_of_samples, n=self.x_prior.shape[0])

    def getPosterior(self, x, y):
        '''
            A posterior is obtained by using prior when conditioned on data. Intuitively, lets says we have fine grid of points or 50 data points between -5 and 5. We build the smooth prior function with mean=0 and covariance k(x,x'). Now when we have additional data points (x) with the target (y) x:y as (2,3), (3,4), (5,7), which says that we are sure that if the input is 2,3,5 then the outputs are 3,4,7 respectively. It is possible that the prior may give a different output y for input 2,3,5. Therefore we condition the priors on the given datapoints. In other words we say, Hey prior smooth function (function approximator) give me the prior function that exactly crosses the three given x points. Posterior ~ P(prior|Datapoints)
            More intuitively lets say we draw infinite samples from the prior, the sample that exactly passes or touched the given datapoints is the posterior
            When we find the posterior, which is a infinite(finite) long function over the distribution of y, for any new unseen input x*, we can easily compute the value of y* using the function


            |param x: training inputs in shape (n,D)
            |param y: training labels in shape (n,D)
        '''
        if x is None and y is None:
            print 'No x and y provided , please provide them'

        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype = float)

        if not isinstance(y, np.ndarray):
            y = np.array(y, dtype = float)

        # Check if the input x and y are numpy ndarray and if they are in of shape (nxD)
        if not x is None:
            print x.ndim
            if x.ndim == 1:
                x = np.reshape(x, (x.shape[0],1))
            self.x = x

        if not y is None:
            print y.ndim
            if y.ndim == 1:
                y = np.reshape(y, (y.shape[0],1))
            self.y = y

        print self.x
        print self.y

        n,d = self.x.shape
        f_mean = self.meanFunc.getMean(a=x)
        self.K = self.covFunc.sq_exp_vect_scipy(a=x, b=x, mode='self') # Squared exponential kernel
        self.L = self.infFunc.cholesky(self.K, n)                                    # Choleski distribution
        self.alpha = solve(self.L.T, solve(self.L,self.y))              # L.T \ (L\y)


    def predict(self, xstar):
        '''
            |param xstar: The test data for which we find the ystar
        '''
        if not isinstance(xstar, np.ndarray):
            xstar = np.array(xstar, dtype = float)

        if not xstar is None:
            if xstar.ndim == 1:
                xstar = np.reshape(xstar, (xstar.shape[0],1))
            self.xstar = xstar

        Kstar_star = self.covFunc.sq_exp_vect_scipy(a=xstar, b=xstar, mode='self')
        Kstar = self.covFunc.sq_exp_vect_scipy(a=self.x, b=xstar, mode='cross')
        
        f_star_mean = np.dot(Kstar.T, self.alpha)             # Formula = mean(xstar) + (kstar.T * inv(k) (f - mean(x))
        v = solve( self.L, Kstar )
        v_f_star = Kstar_star - np.dot(v.T, v)

        self.ystar = f_star_mean.flat
        self.ystar_var = diagonal(v_f_star)




class GPR(GP):      # Gaussian Processes Regression
    '''
        Model for Gaussian Process Regression
    '''
    def __init__(self):
        super(GPR, self).__init__()
        print 'Inside the init of GPR'
        self.meanFunc = Mean.ZeroMean()                      # default prior mean
        self.covFunc = Covariance.SqExp()                    # default prior covariance
        self.infFunc = Inference.Decompose()
        self.sampFunc = Inference.GetSamples()
        # self.likfunc = lik.Gauss()                         # likihood with default noise variance 0.1
        # self.inffunc = inf.Exact()                         # inference method
        # self.optimizer = opt.Minimize(self)                # default optimizer

    def plot_samples(self, x, y, s=1):
        plt.figure()
        for n in range(0,s):
            plt.plot( x, y[:,n] )
        plt.show('hold')


    def plot(self, x, y, xstar):
        ystar = self.ystar
        ystar_var = self.ystar_var
        xstar = np.reshape(xstar,(xstar.shape[0],))
        plt.figure()
        plt.plot(x, y, color=DATACOLOR, ls='None', marker='+',ms=12, mew=2)
        plt.plot(xstar, ystar, color=MEANCOLOR, ls='-', lw=3.)
        plt.fill_between(xstar, ystar + 2.*np.sqrt(ystar_var), ystar - 2.*np.sqrt(ystar_var), facecolor=SHADEDCOLOR,linewidths=0.0)
        plt.grid()
        plt.xlabel('input x')
        plt.ylabel('target y')
        plt.show('hold')






# ns  = xs.shape[0]
# Ks  = covfunc.getCovMatrix(x=x[nz,:], z=xs[id,:], mode='cross')
# Fmu = np.tile(ms,(1,N)) + np.dot(Ks.T,alpha[nz])          # conditional mean fs|f

################################################################################################################################
# '''With Training Data'''


# n, D = x.shape





#####################  ROUGH   ################################

# # The sq exponential kernel covariance matrix is:  
# [[  1.00000000e+00   1.35335283e-01   3.72665317e-06]
#  [  1.35335283e-01   1.00000000e+00   1.11089965e-02]
#  [  3.72665317e-06   1.11089965e-02   1.00000000e+00]]

# # The choleski decomposition witout noise is
# [[  1.41421356e+00   9.56964965e-02   2.63514173e-06]
#  [  0.00000000e+00   1.41097207e+00   7.87311429e-03]
#  [  0.00000000e+00   0.00000000e+00   1.41419165e+00]]


# [[  1.00000050e+00   1.35335216e-01   3.72665131e-06]
#  [  0.00000000e+00   9.90800373e-01   1.12116350e-02]
#  [  0.00000000e+00   0.00000000e+00   9.99937648e-01]]

# [[  1.41421356e+00   9.56964965e-02   2.63514173e-06]
#  [  0.00000000e+00   1.41097207e+00   7.87311429e-03]
#  [  0.00000000e+00   0.00000000e+00   1.41419165e+00]]