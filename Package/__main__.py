import gp as pyGPs
import numpy as np
from gp import GPR


data_dir = '/home/sardendhu/Desktop/StudyHard/Artificial-Intelligence/ALL-Programs/Study/Dataset_n_Models/regression_data.npz'
demoData = np.load(data_dir)
# x = demoData['x']      # training data
# y = demoData['y']      # training target
# z = demoData['xstar']  # test data
# print 'the input data length is: ', z.shape

x = np.array([[1],[3],[6]], dtype = float)
y = np.array([[2],[3],[7]], dtype = float)
# z = np.array([[1.5],[2.66],[4], [4.3], [5], [6], [10]], dtype = float)
print x
model = GPR()      # specify model (GP regression)
print model.meanfunc                    # default prior mean
print model.covfunc                     # default prior covariance
print model.likfunc                     # likihood with default noise variance 0.1
print model.inffunc                     # inference method
print model.optimizer                   # default optimizer

model.getPosterior(x, y) # fit default model (mean zero & rbf kernel) with data
print '__main__ Posterior nlZ done', model.nlZ
print '__main__ Posterior dnlZ done', model.dnlZ
print '__main__ Posterior posterior done', model.posterior
print ''
print '#####################   Running Optimizer    ###########################################################################'
# model.optimize(x, y)     # optimize hyperparamters (default optimizer: single run minimize)
print '#####################   Running function predict  ######################################################################'
# model.predict(z)         # predict test cases
print '#####################   Running Plot data    ###########################################################################'
# model.plot()             # and plot result




# self.meanfunc = mean.Zero()                        # default prior mean
# self.covfunc = cov.RBF()                           # default prior covariance
# self.likfunc = lik.Gauss()                         # likihood with default noise variance 0.1
# self.inffunc = inf.Exact()                         # inference method
# self.optimizer = opt.Minimize(self)                # default optimizer




# class postStruct(object):
#     '''
#     Data structure for posterior

#     | post.alpha: 1d array containing inv(K)*(mu-m), 
#     |             where K is the prior covariance matrix, m the prior mean, 
#     |             and mu the approx posterior mean
#     | post.sW: 1d array containing diagonal of sqrt(W)
#     |          the approximate posterior covariance matrix is inv(inv(K)+W)
#     | post.L : 2d array, L = chol(sW*K*sW+identity(n))

#     Usually, the approximate posterior to be returned admits the form
#     N(mu=m+K*alpha, V=inv(inv(K)+W)), where alpha is a vector and W is diagonal;
#     if not, then L contains instead -inv(K+inv(W)), and sW is unused.
#     '''
#     def __init__(self):
#         self.alpha = np.array([])
#         self.L     = np.array([])
#         self.sW    = np.array([])

#     def __repr__(self):
#         value = "posterior: to get the parameters of the posterior distribution use:\n"+\
#                 "model.posterior.alpha\n"+"model.posterior.L\n"+"model.posterior.sW\n"+\
#         "See documentation and gpml book chapter 2.3 and chapter 3.4.3 for these parameters."
#         return value

#     def __str__(self):
#         value = "posterior distribution described by alpha, sW and L\n"+\
#                 "See documentation and gpml book chapter 2.3 and chapter 3.4.3 for these parameters\n"\
#                 +"alpha:\n"+str(self.alpha)+"\n"+"L:\n"+str(self.L)+"\nsW:\n"+str(self.sW)
#         return value





# class Zero(Mean):
#     '''Zero mean.'''
#     def __init__(self):
#         self.hyp = []
#         self.name = '0'

#     def getMean(self, x=None):
#         n, D = x.shape
#         A = np.zeros((n,1))
#         return A

#     def getDerMatrix(self, x=None, der=None):
#         n, D = x.shape
#         A = np.zeros((n,1))
#         return A

# '''Output: self.hyp = [], self.name = '0' '''

# class RBF(Kernel):
#     '''
#     Squared Exponential kernel with isotropic distance measure. hyp = [log_ell, log_sigma]

#     :param log_ell: characteristic length scale.
#     :param log_sigma: signal deviation.
#     '''
#     def __init__(self, log_ell=0., log_sigma=0.):
#         self.hyp = [log_ell, log_sigma]

#     def getCovMatrix(self,x=None,z=None,mode=None):
#         self.checkInputGetCovMatrix(x,z,mode)
#         ell = np.exp(self.hyp[0])         # characteristic length scale
#         sf2 = np.exp(2.*self.hyp[1])      # signal variance
#         if mode == 'self_test':           # self covariances for the test cases
#             nn,D = z.shape
#             A = np.zeros((nn,1))
#         elif mode == 'train':             # compute covariance matix for training set
#             A = spdist.cdist(old_div(x,ell),old_div(x,ell),'sqeuclidean')
#         elif mode == 'cross':             # compute covariance between data sets x and z
#             A = spdist.cdist(old_div(x,ell),old_div(z,ell),'sqeuclidean')
#         A = sf2 * np.exp(-0.5*A)
#         return A


#     def getDerMatrix(self,x=None,z=None,mode=None,der=None):
#         self.checkInputGetDerMatrix(x,z,mode,der)
#         ell = np.exp(self.hyp[0])         # characteristic length scale
#         sf2 = np.exp(2.*self.hyp[1])      # signal variance
#         if mode == 'self_test':           # self covariances for the test cases
#             nn,D = z.shape
#             A = np.zeros((nn,1))
#         elif mode == 'train':             # compute covariance matix for dataset x
#             A = spdist.cdist(old_div(x,ell),old_div(x,ell),'sqeuclidean')
#         elif mode == 'cross':             # compute covariance between data sets x and z
#             A = spdist.cdist(old_div(x,ell),old_div(z,ell),'sqeuclidean')
#         if der == 0:    # compute derivative matrix wrt 1st parameter
#             A = sf2 * np.exp(-0.5*A) * A
#         elif der == 1:  # compute derivative matrix wrt 2nd parameter
#             A = 2. * sf2 * np.exp(-0.5*A)
#         else:
#             raise Exception("Calling for a derivative in RBF that does not exist")
#         return A


# class Gauss(Likelihood):
#     '''
#     Gaussian likelihood function for regression.

#     :math:`Gauss(t)=\\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-\\frac{(t-y)^2}{2\\sigma^2}}`,
#     where :math:`y` is the mean and :math:`\\sigma` is the standard deviation.

#     hyp = [ log_sigma ]
#     '''
#     def __init__(self, log_sigma=np.log(0.1) ):
#         self.hyp = [log_sigma]

#     def evaluate(self, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
#         from . import inf
#         sn2 = np.exp(2. * self.hyp[0])
#         if inffunc is None:              # prediction mode
#             if y is None:
#                 y = np.zeros_like(mu)
#             s2zero = True
#             if (not s2 is None) and np.linalg.norm(s2) > 0:
#                 s2zero = False
#             if s2zero:                   # log probability
#                 lp = -(y-mu)**2 /sn2/2 - old_div(np.log(2.*np.pi*sn2),2.)
#                 s2 = np.zeros_like(s2)
#             else:
#                 inf_func = inf.EP()   # prediction
#                 lp = self.evaluate(y, mu, s2, inf_func)
#             if nargout>1:
#                 ymu = mu                 # first y moment
#                 if nargout>2:
#                     ys2 = s2 + sn2       # second y moment
#                     return lp,ymu,ys2
#                 else:
#                     return lp,ymu
#             else:
#                 return lp
#         else:
#             if isinstance(inffunc, inf.EP):
#                 if der is None:                                  # no derivative mode
#                     lZ = -(y-mu)**2/(sn2+s2)/2. - old_div(np.log(2*np.pi*(sn2+s2)),2.) # log part function
#                     if nargout>1:
#                         dlZ  = old_div((y-mu),(sn2+s2))                   # 1st derivative w.r.t. mean
#                         if nargout>2:
#                             d2lZ = old_div(-1,(sn2+s2))                   # 2nd derivative w.r.t. mean
#                             return lZ,dlZ,d2lZ
#                         else:
#                            return lZ,dlZ
#                     else:
#                         return lZ
#                 else:                                            # derivative mode
#                     dlZhyp = old_div((old_div((y-mu)**2,(sn2+s2))-1), (1+old_div(s2,sn2))) # deriv. w.r.t. hyp.lik
#                     return dlZhyp
#             elif isinstance(inffunc, inf.Laplace):
#                 if der is None:                                  # no derivative mode
#                     if y is None:
#                         y=0
#                     ymmu = y-mu
#                     lp = old_div(-ymmu**2,(2*sn2)) - old_div(np.log(2*np.pi*sn2),2.)
#                     if nargout>1:
#                         dlp = old_div(ymmu,sn2)                           # dlp, derivative of log likelihood
#                         if nargout>2:                            # d2lp, 2nd derivative of log likelihood
#                             d2lp = old_div(-np.ones_like(ymmu),sn2)
#                             if nargout>3:                        # d3lp, 3rd derivative of log likelihood
#                                 d3lp = np.zeros_like(ymmu)
#                                 return lp,dlp,d2lp,d3lp
#                             else:
#                                 return lp,dlp,d2lp
#                         else:
#                             return lp,dlp
#                     else:
#                         return lp
#                 else:                                            # derivative mode
#                     lp_dhyp   = old_div((y-mu)**2,sn2) - 1                # derivative of log likelihood w.r.t. hypers
#                     dlp_dhyp  = 2*(mu-y)/sn2                     # first derivative,
#                     d2lp_dhyp = 2*np.ones_like(mu)/sn2           # and also of the second mu derivative
#                     return lp_dhyp,dlp_dhyp,d2lp_dhyp
#             '''
#             elif isinstance(inffunc, infVB):
#                 if der is None:
#                     # variational lower site bound
#                     # t(s) = exp(-(y-s)^2/2sn2)/sqrt(2*pi*sn2)
#                     # the bound has the form: b*s - s.^2/(2*ga) - h(ga)/2 with b=y/ga
#                     ga  = s2
#                     n   = len(ga)
#                     b   = y/ga
#                     y   = y*np.ones((n,1))
#                     db  = -y/ga**2 
#                     d2b = 2*y/ga**3
#                     h   = np.zeros((n,1))
#                     dh  = h
#                     d2h = h                           # allocate memory for return args
#                     id  = (ga <= sn2 + 1e-8)          # OK below noise variance
#                     h[id]   = y[id]**2/ga[id] + np.log(2*np.pi*sn2)
#                     h[np.logical_not(id)] = np.inf
#                     dh[id]  = -y[id]**2/ga[id]**2
#                     d2h[id] = 2*y[id]**2/ga[id]**3
#                     id = ga < 0
#                     h[id] = np.inf
#                     dh[id] = 0
#                     d2h[id] = 0                       # neg. var. treatment
#                     varargout = [h,b,dh,db,d2h,d2b]
#                 else:
#                     ga = s2 
#                     n  = len(ga)
#                     dhhyp = np.zeros((n,1))
#                     dhhyp[ga<=sn2] = 2
#                     dhhyp[ga<0] = 0                   # negative variances get a special treatment
#                     varargout = dhhyp                 # deriv. w.r.t. hyp.lik
#             else:
#                 raise Exception('Incorrect inference in lik.Gauss\n')
#         '''





# class Exact(Inference):
#     '''
#     Exact inference for a GP with Gaussian likelihood. Compute a parametrization
#     of the posterior, the negative log marginal likelihood and its derivatives
#     w.r.t. the hyperparameters.
#     '''
#     def __init__(self):
#         self.name = "Exact inference"

#     # likfunc = lik.Gauss()
#     def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
#         if not isinstance(likfunc, lik.Gauss):  # THis checks weather likfunct is an instance of lik.Gauss(), and yes it is
#             raise Exception ('Exact inference only possible with Gaussian likelihood')
#         n, D = x.shape
#         K = covfunc.getCovMatrix(x=x, mode='train')            # evaluate covariance matrix
#         m = meanfunc.getMean(x)                                # evaluate mean vector

#         sn2   = np.exp(2*likfunc.hyp[0])                       # noise variance of likGauss
#         #L     = np.linalg.cholesky(K/sn2+np.eye(n)).T         # Cholesky factor of covariance with noise
#         L     = jitchol(old_div(K,sn2)+np.eye(n)).T            # Cholesky factor of covariance with noise
#         alpha = old_div(solve_chol(L,y-m),sn2)
#         post = postStruct()
#         post.alpha = alpha                                     # return the posterior parameters
#         post.sW    = old_div(np.ones((n,1)),np.sqrt(sn2))               # sqrt of noise precision vector
#         post.L     = L                                         # L = chol(eye(n)+sW*sW'.*K)

#         if nargout>1:                                          # do we want the marginal likelihood?
#             nlZ = old_div(np.dot((y-m).T,alpha),2.) + np.log(np.diag(L)).sum() + n*np.log(2*np.pi*sn2)/2. # -log marg lik
#             if nargout>2:                                      # do we want derivatives?
#                 dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)  # allocate space for derivatives
#                 Q = old_div(solve_chol(L,np.eye(n)),sn2) - np.dot(alpha,alpha.T) # precompute for convenience
#                 dnlZ.lik = [sn2*np.trace(Q)]
#                 if covfunc.hyp:
#                     for ii in range(len(covfunc.hyp)):
#                         dnlZ.cov[ii] = old_div((Q*covfunc.getDerMatrix(x=x, mode='train', der=ii)).sum(),2.)
#                 if meanfunc.hyp:
#                     for ii in range(len(meanfunc.hyp)):
#                         dnlZ.mean[ii] = np.dot(-meanfunc.getDerMatrix(x, ii).T,alpha)
#                         dnlZ.mean[ii] = dnlZ.mean[ii][0,0]
#                 return post, nlZ[0,0], dnlZ
#             return post, nlZ[0,0]
#         return post






# self.name = "Exact inference"