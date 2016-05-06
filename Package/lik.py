import numpy as np
from future.utils import old_div







class Likelihood(object):
    """Base function for Likelihood function"""
    def __init__(self):
        self.hyp = []

    def evaluate(self, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
        '''
        The likelihood functions have two possible modes, the mode being selected
        as follows:


        1) With two or three input arguments:                       [PREDICTION MODE]

         lp = evaluate(y, mu) OR lp, ymu, ys2 = evaluate(y, mu, s2)

            This allows to evaluate the predictive distribution. Let p(y_*|f_*) be the
            likelihood of a test point and N(f_*|mu,s2) an approximation to the posterior
            marginal p(f_*|x_*,x,y) as returned by an inference method. The predictive
            distribution p(y_*|x_*,x,y) is approximated by:
            q(y_*) = \int N(f_*|mu,s2) p(y_*|f_*) df_*

            lp = log( q(y) ) for a particular value of y, if s2 is [] or 0, this
            corresponds to log( p(y|mu) ).

            ymu and ys2 are the mean and variance of the predictive marginal q(y)
            note that these two numbers do not depend on a particular
            value of y.
            All vectors have the same size.


        2) With four or five input arguments, the fouth being an object of class "Inference" [INFERENCE MODE]

         evaluate(y, mu, s2, inf.EP()) OR evaluate(y, mu, s2, inf.Laplace(), i)

         There are two cases for inf, namely a) infLaplace, b) infEP
         The last input i, refers to derivatives w.r.t. the ith hyperparameter.

         | a1)
         | lp,dlp,d2lp,d3lp = evaluate(y, f, [], inf.Laplace()).
         | lp, dlp, d2lp and d3lp correspond to derivatives of the log likelihood.
         | log(p(y|f)) w.r.t. to the latent location f.
         | lp = log( p(y|f) )
         | dlp = d log( p(y|f) ) / df
         | d2lp = d^2 log( p(y|f) ) / df^2
         | d3lp = d^3 log( p(y|f) ) / df^3

         | a2)
         | lp_dhyp,dlp_dhyp,d2lp_dhyp = evaluate(y, f, [], inf.Laplace(), i)
         | returns derivatives w.r.t. to the ith hyperparameter
         | lp_dhyp = d log( p(y|f) ) / (dhyp_i)
         | dlp_dhyp = d^2 log( p(y|f) ) / (df   dhyp_i)
         | d2lp_dhyp = d^3 log( p(y|f) ) / (df^2 dhyp_i)


         | b1)
         | lZ,dlZ,d2lZ = evaluate(y, mu, s2, inf.EP())
         | let Z = \int p(y|f) N(f|mu,s2) df then
         | lZ = log(Z)
         | dlZ = d log(Z) / dmu
         | d2lZ = d^2 log(Z) / dmu^2

         | b2)
         | dlZhyp = evaluate(y, mu, s2, inf.EP(), i)
         | returns derivatives w.r.t. to the ith hyperparameter
         | dlZhyp = d log(Z) / dhyp_i

        Cumulative likelihoods are designed for binary classification. Therefore, they
        only look at the sign of the targets y; zero values are treated as +1.

        Some examples for valid likelihood functions:
         | lik = Gauss([0.1])
         | lik = Erf()
        '''
        pass




class Gauss(Likelihood):
    '''
    Gaussian likelihood function for regression.

    :math:`Gauss(t)=\\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-\\frac{(t-y)^2}{2\\sigma^2}}`,
    where :math:`y` is the mean and :math:`\\sigma` is the standard deviation.

    hyp = [ log_sigma ]
    '''
    def __init__(self, log_sigma=np.log(0.1) ):
        self.hyp = [log_sigma]

    def evaluate(self, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
        import inf
        sn2 = np.exp(2. * self.hyp[0])
        if inffunc is None:              # prediction mode
            if y is None:
                y = np.zeros_like(mu)
            s2zero = True
            if (not s2 is None) and np.linalg.norm(s2) > 0:
                s2zero = False
            if s2zero:                   # log probability
                lp = -(y-mu)**2 /sn2/2 - old_div(np.log(2.*np.pi*sn2),2.)
                s2 = np.zeros_like(s2)
            else:
                inf_func = inf.EP()   # prediction
                lp = self.evaluate(y, mu, s2, inf_func)
            if nargout>1:
                ymu = mu                 # first y moment
                if nargout>2:
                    ys2 = s2 + sn2       # second y moment
                    return lp,ymu,ys2
                else:
                    return lp,ymu
            else:
                return lp
        else:
            if isinstance(inffunc, inf.EP):
                if der is None:                                  # no derivative mode
                    lZ = -(y-mu)**2/(sn2+s2)/2. - old_div(np.log(2*np.pi*(sn2+s2)),2.) # log part function
                    if nargout>1:
                        dlZ  = old_div((y-mu),(sn2+s2))                   # 1st derivative w.r.t. mean
                        if nargout>2:
                            d2lZ = old_div(-1,(sn2+s2))                   # 2nd derivative w.r.t. mean
                            return lZ,dlZ,d2lZ
                        else:
                           return lZ,dlZ
                    else:
                        return lZ
                else:                                            # derivative mode
                    dlZhyp = old_div((old_div((y-mu)**2,(sn2+s2))-1), (1+old_div(s2,sn2))) # deriv. w.r.t. hyp.lik
                    return dlZhyp
            elif isinstance(inffunc, inf.Laplace):
                if der is None:                                  # no derivative mode
                    if y is None:
                        y=0
                    ymmu = y-mu
                    lp = old_div(-ymmu**2,(2*sn2)) - old_div(np.log(2*np.pi*sn2),2.)
                    if nargout>1:
                        dlp = old_div(ymmu,sn2)                           # dlp, derivative of log likelihood
                        if nargout>2:                            # d2lp, 2nd derivative of log likelihood
                            d2lp = old_div(-np.ones_like(ymmu),sn2)
                            if nargout>3:                        # d3lp, 3rd derivative of log likelihood
                                d3lp = np.zeros_like(ymmu)
                                return lp,dlp,d2lp,d3lp
                            else:
                                return lp,dlp,d2lp
                        else:
                            return lp,dlp
                    else:
                        return lp
                else:                                            # derivative mode
                    lp_dhyp   = old_div((y-mu)**2,sn2) - 1                # derivative of log likelihood w.r.t. hypers
                    dlp_dhyp  = 2*(mu-y)/sn2                     # first derivative,
                    d2lp_dhyp = 2*np.ones_like(mu)/sn2           # and also of the second mu derivative
                    return lp_dhyp,dlp_dhyp,d2lp_dhyp