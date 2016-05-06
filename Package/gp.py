import mean
import cov
import lik
import inf
import opt
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from cov import FITCOfKernel
from future.utils import old_div

SHADEDCOLOR = [0.7539, 0.89453125, 0.62890625, 1.0]
MEANCOLOR = [ 0.2109375, 0.63385, 0.1796875, 1.0]
DATACOLOR = [0.12109375, 0.46875, 1., 1.0]

class GP(object):
    '''
    Base class for GP model.
    '''
    def __init__(self):
        super(GP, self).__init__()
        self.usingDefaultMean = True  # was using default mean function now?
        self.meanfunc = None      # mean function
        self.covfunc = None       # covariance function
        self.likfunc = None       # likelihood function
        self.inffunc = None       # inference function
        self.optimizer = None     # optimizer object
        self.nlZ = None           # negative log marginal likelihood
        self.dnlZ = None          # column vector of partial derivatives of the negative
                                  # log marginal likelihood w.r.t. each hyperparameter
        self.posterior = None     # struct representation of the (approximate) posterior
        self.x = None             # n by D matrix of training inputs
        self.y = None             # column vector of length n of training targets
        self.xs = None            # n by D matrix of test inputs
        self.ys = None            # column vector of length nn of true test targets (optional)
        self.ym = None            # column vector (of length ns) of predictive output means
        self.ys2 = None           # column vector (of length ns) of predictive output variances
        self.fm = None            # column vector (of length ns) of predictive latent means
        self.fs2 = None           # column vector (of length ns) of predictive latent variances
        self.lp = None            # column vector (of length ns) of log predictive probabilities


    def getPosterior(self, x=None, y=None, der=True):
        # check wether the number of inputs and labels match
        if x is not None and y is not None: 
            assert x.shape[0] == y.shape[0], "number of inputs and labels does not match"

        # check the shape of inputs
        # transform to the correct shape
        if not x is None:
            if x.ndim == 1:
                x = np.reshape(x, (x.shape[0],1))
            self.x = x

        if not y is None:
            if y.ndim == 1:
                y = np.reshape(y, (y.shape[0],1))
            self.y = y

        if self.usingDefaultMean and self.meanfunc is None:
            c = np.mean(y)
            self.meanfunc = mean.Const(c)    # adapt default prior mean wrt. training labels
            
        # call inference method
        # if isinstance(self.likfunc, lik.Erf): #or is instance(self.likfunc, lik.Logistic):
        #     uy  = unique(self.y)
        #     ind = ( uy != 1 )
        #     if any( uy[ind] != -1):
        #         raise Exception('You attempt classification using labels different from {+1,-1}')

        if not der:
        	print 'gp: There is absolutely no value provided for der '
        	post, nlZ = self.inffunc.evaluate(self.meanfunc, self.covfunc, self.likfunc, self.x, self.y, 2)
        	self.nlZ = nlZ
        	self.posterior = deepcopy(post)
        	return nlZ, post
        else:
        	print 'gp: There is a value provided for der'
        	post, nlZ, dnlZ = self.inffunc.evaluate(self.meanfunc, self.covfunc, self.likfunc, self.x, self.y, 3)
        	print 'gp: the alpha without derivative of hyperparameter is: ', post.alpha
        	print 'gp: the L without derivative of hyperparameter is: ', post.L 
        	print 'gp: the sW without derivative of hyperparameter is: ', post.sW
        	print ''
        	print 'gp: the mean derivative of hyperparameter is: ', dnlZ.mean
        	print 'gp: the cov derivative of hyperparameter is: ', dnlZ.cov 
        	print 'gp: the likelihood derivative of hyperparameter is: ', dnlZ.lik 
        	print ''
        	print 'gp: The NLZ is: ', nlZ
        	self.nlZ       = nlZ
        	self.dnlZ      = deepcopy(dnlZ)
        	self.posterior = deepcopy(post)
        	return nlZ, dnlZ, post



    def optimize(self, x=None, y=None, numIterations=40):
        '''
        Train optimal hyperparameters based on training data,
        adjust new hyperparameters to all mean/cov/lik functions.

        :param x: training inputs in shape (n,D)
        :param y: training labels in shape (n,1)
        '''
        print 'gp: Running the optimize function present inside class GP'
        # check wether the number of inputs and labels match
        if x is not None and y is not None: 
            assert x.shape[0] == y.shape[0], "number of inputs and labels does not match"

        # check the shape of inputs
        # transform to the correct shape
        if not x is None:
        	print 'gp: not x is None'
        	if x.ndim == 1:
        		x = np.reshape(x, (x.shape[0],1))
        	self.x = x
        print 'gp: the value of x is: ', self.x

        if not y is None:
        	print 'gp: not y is None'
        	if y.ndim == 1:
        		y = np.reshape(y, (y.shape[0],1))
        	self.y = y
        print 'gp: the value of x is: ', self.y
        
        if self.usingDefaultMean and self.meanfunc is None:
        	print 'gp: usingDefaultMean and neamfunc are None'
        	c = np.mean(y)
        	self.meanfunc = mean.Const(c)    # adapt default prior mean wrt. training labels

        # optimize
        optimalHyp, optimalNlZ = self.optimizer.findMin(self.x, self.y, numIters = numIterations)
        self.nlZ = optimalNlZ

        # apply optimal hyp to all mean/cov/lik functions here
        self.optimizer._apply_in_objects(optimalHyp)
        self.getPosterior()




    def predict(self, xs, ys=None):
        '''
        Prediction of test points (given by xs) based on training data of the current model.
        This method will output the following value:\n
        predictive output means(ym),\n
        predictive output variances(ys2),\n
        predictive latent means(fm),\n
        predictive latent variances(fs2),\n
        log predictive probabilities(lp).\n
        Theses values can also be achieved from model's property. (e.g. model.ym)

        :param xs: test input in shape of nn by D
        :param ys: test target(optional) in shape of nn by 1 if given

        :return: ym, ys2, fm, fs2, lp
        '''
        # check the shape of inputs
        # transform to correct shape if neccessary
        if xs.ndim == 1:
            xs = np.reshape(xs, (xs.shape[0],1))
        self.xs = xs
        if not ys is None:
            if ys.ndim == 1:
                ys = np.reshape(ys, (ys.shape[0],1))
            self.ys = ys
        print 'gp: predict xs is: ', xs
        print 'gp: predict xs shape is: ', xs.shape
        meanfunc = self.meanfunc
        covfunc  = self.covfunc
        likfunc  = self.likfunc
        inffunc  = self.inffunc
        x = self.x
        y = self.y

        if self.posterior is None:
            self.getPosterior()
        alpha = self.posterior.alpha
        L     = self.posterior.L
        sW    = self.posterior.sW

        nz = list(range(len(alpha[:,0])))         # non-sparse representation
        
        if L == []:                         # in case L is not provided, we compute it
            K = covfunc.getCovMatrix(x=x[nz,:], mode='train')
            #L = np.linalg.cholesky( (np.eye(nz) + np.dot(sW,sW.T)*K).T )
            L = jitchol( (np.eye(len(nz)) + np.dot(sW,sW.T)*K).T )
        Ltril     = np.all( np.tril(L,-1) == 0 ) # is L an upper triangular matrix?
        ns        = xs.shape[0]                  # number of data points
        nperbatch = 1000                         # number of data points per mini batch
        nact      = 0                            # number of already processed test data points
        ymu = np.zeros((ns,1))
        ys2 = np.zeros((ns,1))
        fmu = np.zeros((ns,1))
        fs2 = np.zeros((ns,1))
        lp  = np.zeros((ns,1))
        while nact<=ns-1:                              # process minibatches of test cases to save memory
            id  = list(range(nact,min(nact+nperbatch,ns)))   # data points to process
            print 'gp: The Datapoints to process are: ', id
            kss = covfunc.getCovMatrix(z=xs[id,:], mode='self_test')    # self-variances
            print 'gp: predict: The self covariance of xs (z) is: ', kss
            if isinstance(covfunc, FITCOfKernel): 
            	print 'gp: predict: Yes!! FITCOfkernel is an instance of covfunc'
                Ks = covfunc.getCovMatrix(x=x, z=xs[id,:], mode='cross')   # cross-covariances
                Ks = Ks[nz,:]
            else:
            	print 'gp: predict: NO!! FITCOfkernel is not an instance of covfunc'
            	print 'gp: the x and z used for cross Covariance are: ', x[nz,:], xs[id,:]
                Ks  = covfunc.getCovMatrix(x=x[nz,:], z=xs[id,:], mode='cross')   # cross-covariances
            print 'gp: predict: THe combined Covariance is: ', Ks
            ms  = meanfunc.getMean(xs[id,:])
            print 'gp: The mean (ms) of each data point of (z) is taken as: ',ms 
            N   = (alpha.shape)[1]                     # number of alphas (usually 1; more in case of sampling)
            print 'gp: The value in alpha is (posterior.alpha): ', alpha
            print 'gp: The shape of alpha is: ', N 				# N says How many dimensions 
            Fmu = np.tile(ms,(1,N)) + np.dot(Ks.T,alpha[nz])          # conditional mean fs|f
            print 'gp: THe non-sparse representation (nz) is given as: ', nz
            print 'gp: shape of Ks.T ', Ks.shape
            print 'gp: shape of alpha[nz] ', alpha[nz].shape 
            print 'gp: np.tile(ms,(1,N)) = ', np.tile(ms,(1,N))
            print 'gp: np.dot(Ks.T,alpha[nz] = ', np.dot(Ks.T,alpha[nz])
            print 'gp: THe conditional mean fs|f is: ', Fmu
            fmu[id] = np.reshape(old_div(Fmu.sum(axis=1),N),(len(id),1))       # predictive means
            print 'gp: The new Predicted means are: ', fmu[id]
            if Ltril: # L is triangular => use Cholesky parameters (alpha,sW,L)
            	print sW
            	print np.tile(sW,(1,len(id)))
            	print np.tile(sW,(1,len(id)))*Ks
                V       = np.linalg.solve(L.T,np.tile(sW,(1,len(id)))*Ks)
                print 'gp: The V is: ', V
                fs2[id] = kss - np.array([(V*V).sum(axis=0)]).T             # predictive variances
            else:     # L is not triangular => use alternative parametrization
                fs2[id] = kss + np.array([(Ks*np.dot(L,Ks)).sum(axis=0)]).T # predictive variances
            print 'gp: The predictive variances are: ', fs2[id]
            fs2[id] = np.maximum(fs2[id],0)            # remove numerical noise i.e. negative variances
            print 'gp: The predictive variances after removing numerical noise: ', fs2[id]
            Fs2 = np.tile(fs2[id],(1,N))               # we have multiple values in case of sampling
            print 'gp: Here we sample: ', Fs2 
            if ys is None:
            	print 'gp: The ys is not given DUDE!!'
            	print 'gp: Conditional distribution taken here Fmu[:] is ', Fmu[:]
            	print 'gp: The predictive variance taken here Fmu[:] is ', Fs2[:]
                Lp, Ymu, Ys2 = likfunc.evaluate(None,Fmu[:],Fs2[:],None,None,3)
                print Lp
                print Ymu
                print Ys2
            else:
                Lp, Ymu, Ys2 = likfunc.evaluate(np.tile(ys[id],(1,N)), Fmu[:], Fs2[:],None,None,3)
            lp[id]  = np.reshape( old_div(np.reshape(Lp,(np.prod(Lp.shape),N)).sum(axis=1),N) , (len(id),1) )   # log probability; sample averaging
            print 'gp: lp[id] is: ', lp[id]
            ymu[id] = np.reshape( old_div(np.reshape(Ymu,(np.prod(Ymu.shape),N)).sum(axis=1),N) ,(len(id),1) )  # predictive mean ys|y and ...
            print 'gp: ymu[id] is: ', ymu[id]
            ys2[id] = np.reshape( old_div(np.reshape(Ys2,(np.prod(Ys2.shape),N)).sum(axis=1),N) , (len(id),1) ) # .. variance
            print 'gp: ys2[id] is: ', ys2[id]
            nact = id[-1]+1                  # set counter to index of next data point
        '''Note ymu[id] or lp[id] will only capture only first 1000 data points, as our batch size is only 1000, However ym or lp or ys2'''
        self.ym = ymu
        self.ys2 = ys2
        self.lp = lp
        self.fm = fmu
        self.fs2 = fs2
        if ys is None:
            return ymu, ys2, fmu, fs2, None
        else:
            return ymu, ys2, fmu, fs2, lp

class GPR(GP):
    '''
    Model for Gaussian Process Regression
    '''
    def __init__(self):
        super(GPR, self).__init__()
        self.meanfunc = mean.Zero()                        # default prior mean
        self.covfunc = cov.RBF()                           # default prior covariance
        self.likfunc = lik.Gauss()                         # likihood with default noise variance 0.1
        self.inffunc = inf.Exact()                         # inference method
        self.optimizer = opt.Minimize(self)                # default optimizer


    def setNoise(self,log_sigma):
        '''
        Set noise other than default noise value

        :param log_sigma: logarithm of the noise sigma
        '''
        self.likfunc = lik.Gauss(log_sigma)


    def plot(self,axisvals=None):
        '''
        Plot 1d GP regression result.

        :param list axisvals: [min_x, max_x, min_y, max_y] setting the plot range
        '''
        xs = self.xs    # test point
        x = self.x
        y = self.y
        ym = self.ym    # predictive test mean
        ys2 = self.ys2  # predictive test variance
        print 'x axis for train data is: ', x
        print 'y axis for train data is: ', y
        print 'x axis for test data is: ', xs
        print 'predictive mean ym is: ', ym
        print 'predictive variance is: ', ys2
        # print 'y axis for test data is: ', 
        plt.figure()
        xss  = np.reshape(xs,(xs.shape[0],))
        ymm  = np.reshape(ym,(ym.shape[0],))
        ys22 = np.reshape(ys2,(ys2.shape[0],))
        plt.plot(x, y, color=DATACOLOR, ls='None', marker='+',ms=12, mew=2)
        plt.plot(xs, ym, color=MEANCOLOR, ls='-', lw=3.)
        plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=SHADEDCOLOR,linewidths=0.0)
        plt.grid()
        if not axisvals is None:
            plt.axis(axisvals)
        plt.xlabel('input x')
        plt.ylabel('target y')
        plt.show()