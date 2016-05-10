
from numpy.random import seed
from infpy.gp import GaussianProcess, gp_1D_X_range, gp_plot_samples_from
from pylab import plot, savefig, title, close, figure, xlabel, ylabel
# seed RNG to make reproducible and close all existing plot windows
seed(2)
close(’all’)
#
# Kernel
#
from infpy.gp import SquaredExponentialKernel as SE

kernel = SE([1])
#
# Part of X-space we will plot samples from
#
support = gp_1D_X_range(-10.0, 10.01, .125)
#
# Plot samples from prior
#
figure()
gp = GaussianProcess([], [], kernel)
gp_plot_samples_from(gp, support, num_samples=3)
xlabel('x')
ylabel('f(x)'')
title('Samples from the prior')
savefig('samples_from_prior.png')
savefig('samples_from_prior.eps')
#
# Data
#
9
X = [[-5.], [-2.], [3.], [3.5]]
Y = [2.5, 2, -.5, 0.]
#
# Plot samples from posterior
#
figure()
plot([x[0] for x in X], Y, 'ks')
gp = GaussianProcess(X, Y, kernel)
gp_plot_samples_from(gp, support, num_samples=3)
xlabel('x')
ylabel('f(x)'')
title('Samples from the posterior')
savefig('samples_from_posterior.png')
savefig('samples_from_posterior.eps')






from numpy.random import seed
from infpy.gp import GaussianProcess, gp_1D_X_range, gp_plot_samples_from
from pylab import plot, savefig, title, close, figure, xlabel, ylabel
from infpy.gp import SquaredExponentialKernel as SE
from infpy.gp import Matern52Kernel as Matern52
from infpy.gp import Matern52Kernel as Matern32
from infpy.gp import RationalQuadraticKernel as RQ
from infpy.gp import NeuralNetworkKernel as NN
from infpy.gp import FixedPeriod1DKernel as Periodic
from infpy.gp import noise_kernel as noise





class GaussianProcess( object ):
    """
    A Gaussian process.
    Following notation in section 2.2 of `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 
    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    """

    def __init__( self, X, y, k ):
        """Initialise Gaussian process
        X: training data points
        y: training outputs
        k: covariance function
        """
        self.k = k
        self.reestimate( X, y )

    def reestimate( self, X, y ):
        """
        Reestimate process given the data
        """
        from numpy import matrix
        if len( X ) != len( y ):
            raise RuntimeError( """Supplied %d data and %d target values. """
                    """These must be equal.""" % ( len(X), len(y) ) )
        self.X = X
        self.y = matrix( y ).T
        self.n = self.y.shape[0]
        self._update()


    def predict( self, x_star ):
        """
        Predict the process's values on the input values
        @arg x_star: Prediction points
        @return: ( mean, variance, LL )
        where mean are the predicted means, variance are the predicted
        variances and LL is the log likelihood of the data for the given
        value of the parameters (i.e. not integrating over hyperparameters)
        """
        from numpy.linalg import solve
        import types
        # print 'Predicting'
        if 0 == len(self.X):
            f_star_mean = matrix(zeros((len(x_star),1), numpy.float64))
            v = matrix(zeros((0,len(x_star)), numpy.float64))
        else:
            k_star = self.calc_covariance( self.X, x_star )
            f_star_mean = k_star.T * self._alpha
            if 0 == len(x_star): # no training data
                v = matrix(zeros((0,len(x_star)), numpy.float64))
            else:
                v = solve( self._L, k_star )
        V_f_star = self.calc_covariance(x_star) - v.T * v
        # print 'Done predicting'
        #import IPython; IPython.Debugger.Pdb().set_trace()
        return ( f_star_mean, V_f_star, self.LL )

    def calc_covariance( self, x1, x2 = None ):
        """
        Calculate covariance matrix of x1 against x2
        if x2 is None calculate symmetric matrix
        """
        import types
        symmetric = None == x2 # is it symmetric?
        if symmetric: x2 = x1
        def f(i1,i2): return self.k( x1[i1], x2[i2], symmetric and i1 == i2 )
        return \
                infpy.matrix_from_function(
                        f,
                        ( len( x1 ), len( x2 ) ),
                        numpy.float64,
                        symmetric )

def gp_plot_samples_from(
        gp,
        support,
        num_samples=1
):
    """
    Plot samples from a Gaussian process.
    """
    from pylab import plot
    mean, sigma, LL = gp.predict(support)
    gp_plot_prediction(support, mean, sigma)
    for i in xrange(num_samples):
        sample = numpy.random.multivariate_normal(
            asarray(mean).reshape(len(support),),
            sigma
        )
        plot([x[0] for x in support], sample)



def gp_plot_prediction(
        predict_x,
        mean,
        variance = None
):
    """
    Plot a gp's prediction using pylab including error bars if variance specified
    Error bars are 2 * standard_deviation as in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 
    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    """
    from pylab import plot, concatenate, fill
    if None != variance:
        # check variances are just about +ve - could signify a bug if not
        assert diagonal( variance ).all() > -1e-10
        data = [
                (x,y,max(v,0.0))
                for x,y,v
                in zip( predict_x, mean.flat, diagonal( variance ) )
        ]
    else:
        data = [
                (x,y)
                for x,y
                in zip( predict_x, mean )
        ]
    data.sort( key = lambda d: d[0] ) # sort on X axis
    predict_x = [ d[0] for d in data ]
    predict_y = asarray( [ d[1] for d in data ] )
    plot( predict_x, predict_y, color='k', linestyle=':' )
    if None != variance:
        sd = sqrt( asarray( [ d[2] for d in data ] ) )
        var_x = concatenate((predict_x, predict_x[::-1]))
        var_y = concatenate((predict_y + 2.0 * sd, (predict_y - 2.0 * sd)[::-1]))
        p = fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3')
        #plot( predict_x, predict_y + 2.0 * sd, 'k--' )
        #plot( predict_x, predict_y - 2.0 * sd, 'k--' )    