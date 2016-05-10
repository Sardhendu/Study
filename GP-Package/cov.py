import numpy as np

from future.utils import old_div

import math
import scipy.spatial.distance as spdist



class Kernel(object):
    """
    This is a base class of Kernel functions
    there is no computation in this class, it just defines rules about a kernel class should have
    each covariance function will inherit it and implement its own behaviour
    """
    def __init__(self):
        self.hyp = []
        self.para = []

    def __repr__(self):
        strvalue =str(type(self))+': to get the kernel matrix or kernel derviatives use: \n'+\
          'model.covfunc.getCovMatrix()\n'+\
          'model.covfunc.getDerMatrix()'
        return strvalue

    def getCovMatrix(self,x=None,z=None,mode=None):
        '''
        Return the specific covariance matrix according to input mode

        :param x: training data
        :param z: test data
        :param str mode: 'self_test' return self covariance matrix of test data(test by 1).
                         'train' return training covariance matrix(train by train).
                         'cross' return cross covariance matrix between x and z(train by test)

        :return: the corresponding covariance matrix
        '''
        pass

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        '''
        Compute derivatives wrt. hyperparameters according to input mode

        :param x: training data
        :param z: test data
        :param str mode: 'self_test' return self derivative matrix of test data(test by 1).
                         'train' return training derivative matrix(train by train).
                         'cross' return cross derivative matrix between x and z(train by test)
        :param int der: index of hyperparameter whose derivative to be computed

        :return: the corresponding derivative matrix
        '''
        pass

    def checkInputGetCovMatrix(self,x,z,mode):
        '''
        Check validity of inputs for the method getCovMatrix()

        :param x: training data
        :param z: test data
        :param str mode: 'self_test' return self derivative matrix of test data(test by 1).
                         'train' return training derivative matrix(train by train).
                         'cross' return cross derivative matrix between x and z(train by test)
        '''
        if mode is None:
            raise Exception("Specify the mode: 'train' or 'cross'")
        if x is None and z is None:
            raise Exception("Specify at least one: training input (x) or test input (z) or both.")
        if mode == 'cross':
            if x is None or z is None:
                raise Exception("Specify both: training input (x) and test input (z) for cross covariance.")

    def checkInputGetDerMatrix(self,x,z,mode,der):
        '''
        Check validity of inputs for the method getDerMatrix()

        :param x: training data
        :param z: test data
        :param str mode: 'self_test' return self derivative matrix of test data(test by 1).
                         'train' return training derivative matrix(train by train).
                         'cross' return cross derivative matrix between x and z(train by test)
        :param int der: index of hyperparameter whose derivative to be computed
        '''
        if mode is None:
            raise Exception("Specify the mode: 'train' or 'cross'")
        if x is None and z is None:
            raise Exception("Specify at least one: training input (x) or test input (z) or both.")
        if mode == 'cross':
            if x is None or z is None:
                raise Exception("Specify both: training input (x) and test input (z) for cross covariance.")
        if der is None:
            raise Exception("Specify the index of parameters of the derivatives.")

    # overloading
    def __add__(self,cov):
        '''
        Overloading + operator.

        :param cov: covariance function
        :return: an instance of SumOfKernel
        '''
        return SumOfKernel(self,cov)

    # overloading
    def __mul__(self,other):
        '''
        Overloading * operator.
        Using * for both multiplication with scalar and product of kernels
        depending on the type of the two objects.

        :param other: covariance function as product or int/float as scalar
        :return: an instance of ScaleOfKernel or ProductOfKernel
        '''
        if isinstance(other, int) or isinstance(other, float):
            return ScaleOfKernel(self,other)
        elif isinstance(other, Kernel):
            return ProductOfKernel(self,other)
        else:
            print("only numbers and Kernels are supported operand types for *")

    # overloading
    __rmul__ = __mul__

    def fitc(self,inducingInput):
        '''
        Covariance function to be used together with the FITC approximation.
        Setting FITC gp model will implicitly call this method.

        :return: an instance of FITCOfKernel
        '''
        return FITCOfKernel(self,inducingInput)

    # can be replaced by spdist from scipy
    def _sq_dist(self, a, b=None):
        '''Compute a matrix of all pairwise squared distances
        between two sets of vectors, stored in the row of the two matrices:
        a (of size n by D) and b (of size m by D).'''
        n = a.shape[0]
        D = a.shape[1]
        m = n
        if b is None:
            b = a.transpose()
        else:
            m = b.shape[0]
            b = b.transpose()
        C = np.zeros((n,m))
        for d in range(0,D):
            tt  = a[:,d]
            tt  = tt.reshape(n,1)
            tem = np.kron(np.ones((1,m)), tt)
            tem = tem - np.kron(np.ones((n,1)), b[d,:])
            C   = C + tem * tem
        return C



class RBF(Kernel):
    '''
    Squared Exponential kernel with isotropic distance measure. hyp = [log_ell, log_sigma]

    :param log_ell: characteristic length scale.
    :param log_sigma: signal deviation.
    '''
    def __init__(self, log_ell=0., log_sigma=0.):
        self.hyp = [log_ell, log_sigma]                 # HyperParameter for RBF Kernel (squared exponential kernel)
        print 'Cov.RBF!! hyp: ', self.hyp

    def getCovMatrix(self,x=None,z=None,mode=None):
        self.checkInputGetCovMatrix(x,z,mode)
        ell = np.exp(self.hyp[0])         # characteristic length scale
        sf2 = np.exp(2.*self.hyp[1])      # signal variance
        print 'cov: the value of ell is: ', ell
        print 'cov: the value of sf2 is: ', sf2
        if mode == 'self_test':           # self covariances for the test cases
            print 'cov. You are accessing thr test covariance of class RBF kernel' 
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for training set
            print 'cov. You are accessing thr train covariance of class RBF kernel'  
            # print 'cov: the old_dic value is: ', old_div(x,ell)
            A = spdist.cdist(old_div(x,ell),old_div(x,ell),'sqeuclidean')
            # A = spdist.cdist(x,x,'sqeuclidean')
            # print 'cov: The covariance matrix A is: ', A
        elif mode == 'cross':             # compute covariance between data sets x and z
            print 'cov. You are accessing thr cross covariance of class RBF kernel'  
            A = spdist.cdist(old_div(x,ell),old_div(z,ell),'sqeuclidean')
        A = sf2 * np.exp(-0.5*A)
        return A


    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        self.checkInputGetDerMatrix(x,z,mode,der)
        ell = np.exp(self.hyp[0])         # characteristic length scale
        sf2 = np.exp(2.*self.hyp[1])      # signal variance
        if mode == 'self_test':           # self covariances for the test cases
            nn,D = z.shape
            A = np.zeros((nn,1))
        elif mode == 'train':             # compute covariance matix for dataset x
            A = spdist.cdist(old_div(x,ell),old_div(x,ell),'sqeuclidean')
        elif mode == 'cross':             # compute covariance between data sets x and z
            A = spdist.cdist(old_div(x,ell),old_div(z,ell),'sqeuclidean')
        if der == 0:    # compute derivative matrix wrt 1st parameter
            A = sf2 * np.exp(-0.5*A) * A
        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * np.exp(-0.5*A)
        else:
            raise Exception("Calling for a derivative in RBF that does not exist")
        return A






class FITCOfKernel(Kernel):
    '''
    Covariance function to be used together with the FITC approximation.
    The function allows for more than one output argument and does not respect the
    interface of a proper covariance function.
    Instead of outputing the full covariance, it returns cross-covariances between
    the inputs x, z and the inducing inputs xu as needed by infFITC
    '''
    def __init__(self,cov,inducingInput):
        self.inducingInput = inducingInput
        self.covfunc = cov
        self._hyp = cov.hyp

    def _getHyp(self):
        return self._hyp
    def _setHyp(self, hyp):
        self._hyp = hyp
        self.covfunc.hyp = hyp
    hyp = property(_getHyp,_setHyp)

    def getCovMatrix(self,x=None,z=None,mode=None):
        self.checkInputGetCovMatrix(x,z,mode)
        xu = self.inducingInput
        if not x is None:
            try:
                assert(xu.shape[1] == x.shape[1])
            except AssertionError:
                raise Exception('Dimensionality of inducing inputs must match training inputs')
        if mode == 'self_test':           # self covariances for the test cases
            K = self.covfunc.getCovMatrix(z=z,mode='self_test')
            return K
        elif mode == 'train':             # compute covariance matix for training set
            K   = self.covfunc.getCovMatrix(z=x,mode='self_test')
            Kuu = self.covfunc.getCovMatrix(x=xu,mode='train')
            Ku  = self.covfunc.getCovMatrix(x=xu,z=x,mode='cross')
            return K, Kuu, Ku
        elif mode == 'cross':             # compute covariance between data sets x and z
            K = self.covfunc.getCovMatrix(x=xu,z=z,mode='cross')
            return K

    def getDerMatrix(self,x=None,z=None,mode=None,der=None):
        self.checkInputGetDerMatrix(x,z,mode,der)
        xu = self.inducingInput
        if not x is None:
            try:
                assert(xu.shape[1] == x.shape[1])
            except AssertionError:
                raise Exception('Dimensionality of inducing inputs must match training inputs')
        if mode == 'self_test':           # self covariances for the test cases
            K = self.covfunc.getDerMatrix(z=z,mode='self_test',der=der)
            return K
        elif mode == 'train':             # compute covariance matix for training set
            K   = self.covfunc.getDerMatrix(z=x,mode='self_test',der=der)
            Kuu = self.covfunc.getDerMatrix(x=xu,mode='train',der=der)
            Ku  = self.covfunc.getDerMatrix(x=xu,z=x,mode='cross',der=der)
            return K, Kuu, Ku
        elif mode == 'cross':             # compute covariance between data sets x and z
            K = self.covfunc.getDerMatrix(x=xu,z=z,mode='cross',der=der)
            return K