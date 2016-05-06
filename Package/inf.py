import lik
import numpy as np
from future.utils import old_div
from tools import jitchol, solve_chol


class postStruct(object):
    '''
    Data structure for posterior

    | post.alpha: 1d array containing inv(K)*(mu-m), 
    |             where K is the prior covariance matrix, m the prior mean, 
    |             and mu the approx posterior mean
    | post.sW: 1d array containing diagonal of sqrt(W)
    |          the approximate posterior covariance matrix is inv(inv(K)+W)
    | post.L : 2d array, L = chol(sW*K*sW+identity(n))

    Usually, the approximate posterior to be returned admits the form
    N(mu=m+K*alpha, V=inv(inv(K)+W)), where alpha is a vector and W is diagonal;
    if not, then L contains instead -inv(K+inv(W)), and sW is unused.
    '''
    def __init__(self):
        self.alpha = np.array([])
        self.L     = np.array([])
        self.sW    = np.array([])

    def __repr__(self):
        value = "posterior: to get the parameters of the posterior distribution use:\n"+\
                "model.posterior.alpha\n"+"model.posterior.L\n"+"model.posterior.sW\n"+\
        "See documentation and gpml book chapter 2.3 and chapter 3.4.3 for these parameters."
        return value

    def __str__(self):
        value = "posterior distribution described by alpha, sW and L\n"+\
                "See documentation and gpml book chapter 2.3 and chapter 3.4.3 for these parameters\n"\
                +"alpha:\n"+str(self.alpha)+"\n"+"L:\n"+str(self.L)+"\nsW:\n"+str(self.sW)
        return value


class Inference(object):
    '''
    Base class for inference. Defined several tool methods in it.
    '''
    def __init__(self):
        pass

    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        '''
        Inference computation based on inputs.
        post, nlZ, dnlZ = inf.evaluate(mean, cov, lik, x, y)

            | INPUT:
            | cov: name of the covariance function (see covFunctions.m)
            | lik: name of the likelihood function (see likFunctions.m)
            | x: n by D matrix of training inputs
            | y: 1d array (of size n) of targets

            | OUTPUT:
            | post(postStruct): struct representation of the (approximate) posterior containing:
            | nlZ: returned value of the negative log marginal likelihood
            | dnlZ(dnlZStruct): struct representation for derivatives of the negative log marginal likelihood
            | w.r.t. each hyperparameter.

        Usually, the approximate posterior to be returned admits the form:
        N(m=K*alpha, V=inv(inv(K)+W)), where alpha is a vector and W is diagonal;
        if not, then L contains instead -inv(K+inv(W)), and sW is unused.

        For more information on the individual approximation methods and their
        implementations, see the respective inference function below. See also gp.py

        :param meanfunc: mean function
        :param covfunc: covariance function
        :param likfunc: likelihood function
        :param x: training data
        :param y: training labels
        :param nargout: specify the number of output(1,2 or 3)
        :return: posterior, negative-log-marginal-likelihood, derivative for negative-log-marginal-likelihood-likelihood
        '''
        pass



class dnlZStruct(object):
    '''
    Data structure for the derivatives of mean, cov and lik functions.

    |dnlZ.mean: list of derivatives for each hyperparameters in mean function
    |dnlZ.cov: list of derivatives for each hyperparameters in covariance function
    |dnlZ.lik: list of derivatives for each hyperparameters in likelihood function
    '''
    def __init__(self, m, c, l):
        self.mean = []
        self.cov = []
        self.lik = []
        if m.hyp:
        	self.mean = [0 for i in range(len(m.hyp))]
        	print 'inf: The hyperparamenters of the cov matrix are: ', m.hyp
        	print 'inf: Yes the Covariance matrix has hyperparamenters, their derivatives are: ', self.mean
        if c.hyp:
        	self.cov  = [0 for i in range(len(c.hyp))]
        	print 'inf: The hyperparamenters of the cov matrix are: ', c.hyp
        	print 'inf: Yes the Covariance matrix has hyperparamenters, their derivatives are: ', self.cov
        if l.hyp:
        	self.lik  = [0 for i in range(len(l.hyp))]
        	print 'inf: The hyperparamenters of the Cholesky decomposition matrix are: ', l.hyp
        	print 'inf: Yes the Cholesky decomposition matrix has hyperparamenters, their derivatives are: ', self.lik

    def __str__(self):
        value = "Derivatives of mean, cov and lik functions:\n" +\
                "mean:"+str(self.mean)+"\n"+\
                "cov:"+str(self.cov)+"\n"+\
                "lik:"+str(self.lik)
        return value

    def __repr__(self):
        value = "dnlZ: to get the derivatives of mean, cov and lik functions use:\n" +\
                "model.dnlZ.mean\n"+"model.dnlZ.cov\n"+"model.dnlZ.lik"
        return value



class Exact(Inference):
    '''
    Exact inference for a GP with Gaussian likelihood. Compute a parametrization
    of the posterior, the negative log marginal likelihood and its derivatives
    w.r.t. the hyperparameters.
    '''
    def __init__(self):
        self.name = "Exact inference"
        print 'inf: The name of function is: ', self.name

    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        if not isinstance(likfunc, lik.Gauss):
            raise Exception ('Exact inference only possible with Gaussian likelihood')
        n, D = x.shape
        print 'inf: The x is: ', x
        K = covfunc.getCovMatrix(x=x, mode='train')            # evaluate covariance matrix
        print 'inf: The squared exponential kernel covariance matrix is: ', K
        m = meanfunc.getMean(x)                                # evaluate mean vector
        print 'inf: The mean is: ', m
        sn2   = np.exp(2*likfunc.hyp[0])                       # noise variance of likGauss
        print 'inf: The Gaussian with added noise is: ', sn2
        # L     = np.linalg.cholesky(K/sn2+np.eye(n)).T         # Cholesky factor of covariance with noise
        L     = jitchol(old_div(K,sn2)+np.eye(n)).T            # Cholesky factor of covariance with noise
        # L     = jitchol(K+np.eye(n)).T            # Cholesky factor of covariance without noise
        print 'inf: The choleski decomposition matrix without noise is: ', L
        alpha = old_div(solve_chol(L,y-m),sn2)				   # With Noise
        # alpha = solve_chol(L,y-m)							   # Without Noise
        print 'inf: The alpha matrix without noise is: ', alpha
        post = postStruct()
        print 'inf: the alpha before posterior process is: ', post.alpha
        print 'inf: the L before posterior process is: ', post.L 
        print 'inf: the sW before posterior process is: ', post.sW 
        post.alpha = alpha                                     # return the posterior parameters
        print np.ones((n,1))
        print np.sqrt(sn2)
        # post.sW    = old_div(np.ones(n,1))      # Without noise  equivellent to a/b
        post.sW    = old_div(np.ones((n,1)),np.sqrt(sn2))      # sqrt of noise precision vector   equivellent to a/b
        print 'inf: sW is: ', post.sW
        post.L     = L                                         # L = chol(eye(n)+sW*sW'.*K)

        if nargout>1:                                          # do we want the marginal likelihood?
        	nlZ = old_div(np.dot((y-m).T,alpha),2.) + np.log(np.diag(L)).sum() + n*np.log(2*np.pi*sn2)/2. # -log marg lik
        	if nargout>2:                                      # do we want derivatives?
        		dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)  # allocate space for derivatives
        		Q = old_div(solve_chol(L,np.eye(n)),sn2) - np.dot(alpha,alpha.T) # precompute for convenience
        		dnlZ.lik = [sn2*np.trace(Q)]
        		if covfunc.hyp:
        			for ii in range(len(covfunc.hyp)):
        				dnlZ.cov[ii] = old_div((Q*covfunc.getDerMatrix(x=x, mode='train', der=ii)).sum(),2.)
        		if meanfunc.hyp:
        			for ii in range(len(meanfunc.hyp)):
        				dnlZ.mean[ii] = np.dot(-meanfunc.getDerMatrix(x, ii).T,alpha)
        				dnlZ.mean[ii] = dnlZ.mean[ii][0,0]
        		return post, nlZ[0,0], dnlZ
        	return post, nlZ[0,0]
        return post


class EP(Inference):
    '''
    Expectation Propagation approximation to the posterior Gaussian Process.
    '''
    def __init__(self):
        self.name = 'Expectation Propagation'
        self.last_ttau = None
        self.last_tnu = None
    def evaluate(self, meanfunc, covfunc, likfunc, x, y, nargout=1):
        tol = 1e-4; max_sweep = 10; min_sweep = 2 # tolerance to stop EP iterations
        n = x.shape[0]
        inffunc = self
        K = covfunc.getCovMatrix(x=x, mode='train') # evaluate the covariance matrix
        m = meanfunc.getMean(x)                   # evaluate the mean vector
        nlZ0 = -likfunc.evaluate(y, m, np.reshape(np.diag(K),(np.diag(K).shape[0],1)), inffunc).sum()
        if self.last_ttau is None:                # find starting point for tilde parameters
            ttau  = np.zeros((n,1))               # initialize to zero if we have no better guess
            tnu   = np.zeros((n,1))
            Sigma = K                             # initialize Sigma and mu, the parameters of ..
            mu    = np.zeros((n,1))               # .. the Gaussian posterior approximation
            nlZ   = nlZ0
        else:
            ttau = self.last_ttau                 # try the tilde values from previous call
            tnu  = self.last_tnu
            Sigma, mu, nlZ, L = self._epComputeParams(K, y, ttau, tnu, likfunc, m, inffunc)
            if nlZ > nlZ0:                        # if zero is better ..
                ttau = np.zeros((n,1))            # .. then initialize with zero instead
                tnu  = np.zeros((n,1))
                Sigma = K                         # initialize Sigma and mu, the parameters of ..
                mu = np.zeros((n,1))              # .. the Gaussian posterior approximation
                nlZ = nlZ0
        nlZ_old = np.inf; sweep = 0               # converged, max. sweeps or min. sweeps?
        while (np.abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or (sweep < min_sweep):
            nlZ_old = nlZ; sweep += 1
            rperm = range(n)                     # randperm(n)
            for ii in rperm:                      # iterate EP updates (in random order) over examples
                tau_ni = old_div(1,Sigma[ii,ii]) - ttau[ii]#  first find the cavity distribution ..
                nu_ni  = old_div(mu[ii],Sigma[ii,ii]) + m[ii]*tau_ni - tnu[ii]    # .. params tau_ni and nu_ni
                # compute the desired derivatives of the indivdual log partition function
                lZ,dlZ,d2lZ = likfunc.evaluate(y[ii], old_div(nu_ni,tau_ni), old_div(1,tau_ni), inffunc, None, 3)
                ttau_old = copy(ttau[ii])         # then find the new tilde parameters, keep copy of old
                ttau[ii] = old_div(-d2lZ,(1.+old_div(d2lZ,tau_ni)))
                ttau[ii] = max(ttau[ii],0)        # enforce positivity i.e. lower bound ttau by zero
                tnu[ii]  = old_div(( dlZ + (m[ii]-old_div(nu_ni,tau_ni))*d2lZ ),(1.+old_div(d2lZ,tau_ni)))
                ds2 = ttau[ii] - ttau_old         # finally rank-1 update Sigma ..
                si  = np.reshape(Sigma[:,ii],(Sigma.shape[0],1))
                Sigma = Sigma - ds2/(1.+ds2*si[ii])*np.dot(si,si.T)   # takes 70# of total time
                mu = np.dot(Sigma,tnu)                                # .. and recompute mu
            # recompute since repeated rank-one updates can destroy numerical precision
            Sigma, mu, nlZ, L = self._epComputeParams(K, y, ttau, tnu, likfunc, m, inffunc)
        if sweep == max_sweep:
            pass
            # print '[warning] maximum number of sweeps reached in function infEP'
        self.last_ttau = ttau; self.last_tnu = tnu          # remember for next call
        sW = np.sqrt(ttau); alpha = tnu-sW*solve_chol(L,sW*np.dot(K,tnu))
        post = postStruct()
        post.alpha = alpha                                  # return the posterior params
        post.sW    = sW
        post.L     = L
        if nargout>2:                                       # do we want derivatives?
            dnlZ = dnlZStruct(meanfunc, covfunc, likfunc)   # allocate space for derivatives
            ssi  = np.sqrt(ttau)
            V = np.linalg.solve(L.T,np.tile(ssi,(1,n))*K)
            Sigma = K - np.dot(V.T,V)
            mu = np.dot(Sigma,tnu)
            Dsigma = np.reshape(np.diag(Sigma),(np.diag(Sigma).shape[0],1))
            tau_n = old_div(1,Dsigma)-ttau                           # compute the log marginal likelihood
            nu_n  = old_div(mu,Dsigma)-tnu                           # vectors of cavity parameters
            F = np.dot(alpha,alpha.T) - np.tile(sW,(1,n))* \
                solve_chol(L,np.diag(np.reshape(sW,(sW.shape[0],))))   # covariance hypers
            for jj in range(len(covfunc.hyp)):
                dK = covfunc.getDerMatrix(x=x, mode='train', der=jj)
                dnlZ.cov[jj] = old_div(-(F*dK).sum(),2.)
            for ii in range(len(likfunc.hyp)):
                dlik = likfunc.evaluate(y, old_div(nu_n,tau_n), old_div(1,tau_n), inffunc, ii)
                dnlZ.lik[ii] = -dlik.sum()
            junk,dlZ = likfunc.evaluate(y, old_div(nu_n,tau_n), old_div(1,tau_n), inffunc, None, 2) # mean hyps
            for ii in range(len(meanfunc.hyp)):
                dm = meanfunc.getDerMatrix(x, ii)
                dnlZ.mean[ii] = -np.dot(dlZ.T,dm)
                dnlZ.mean[ii] = dnlZ.mean[ii][0,0]
            return post, nlZ[0], dnlZ
        else:
            return post, nlZ[0]
