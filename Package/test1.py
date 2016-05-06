
class Zero(Mean):
    '''Zero mean.'''
    def __init__(self):
        self.hyp = []
        self.name = '0'

    def getMean(self, x=None):
        n, D = x.shape
        A = np.zeros((n,1))
        return A

    def getDerMatrix(self, x=None, der=None):
        n, D = x.shape
        A = np.zeros((n,1))
        return A

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
