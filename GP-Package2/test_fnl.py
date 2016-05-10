import numpy as np
import numpy
import scipy.spatial.distance as spdist
import math
from numpy import matrix, array, zeros, log, diagonal, trace, arange, asarray, mean, sqrt
from pylab import plot, savefig, title, close, figure, xlabel, ylabel

def gp_1D_X_range(xmin, xmax, step=1.):
    """
    @return: An array of points between xmin and xmax with the given step size. The shape of the array
    will be (N, 1) where N is the number of points as this is the format the GP code expects.
    Implementation uses numpy.arange so last element of array may be > xmax.
    """
    result = numpy.arange(xmin, xmax, step)
    result.resize((len(result), 1))
    return result


class GaussianProcess( object ):
    """
    A Gaussian process.
    Following notation in section 2.2 of `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 
    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    """

    def __init__( self, X, y):
        """Initialise Gaussian process
        X: training data points
        y: training outputs
        k: covariance function
        """
        self.reestimate( X, y )

    def reestimate( self, X, y ):
        """
        Reestimate process given the data
        """
        print ''
        print 'In the Reestimate function ................................................'
        from numpy import matrix
        if len( X ) != len( y ):
            raise RuntimeError( """Supplied %d data and %d target values. """
                    """These must be equal.""" % ( len(X), len(y) ) )
        print 'The X passed is: ', X
        print 'The Y passes is: ', y
        self.X = X
        self.y = matrix( y ).T
        self.n = self.y.shape[0]
        self._update()



        # '''
        # Calculate those terms for prediction that do not depend on predictive
        # inputs.'''


        # print ''
        # print 'In the Update Function ...........................................................'

    def _update( self ):
        """
        Calculate those terms for prediction that do not depend on predictive
        inputs.
        """
        print ''
        print 'In the Update function ......................................'
        from numpy.linalg import cholesky, solve, LinAlgError
        from numpy import transpose, eye, matrix
        import types
        self._K = self.calc_covariance( self.X )
        print 'The Coavriance matrix for the train set is: ', self._K
        if not self._K.shape[0]: # we didn't have any data
            print 'I am in the if condition of the update'
            self._L = matrix(zeros((0,0), numpy.float64))
            self._alpha = matrix(zeros((0,1), numpy.float64))
            self.LL = 0.
        else:
            print 'I am in the else condition of the update:'
            try:
                self._L = matrix(cholesky(self._K))
                print 'The Choleski is: ', self._L
            except LinAlgError, detail:
                raise RuntimeError( """Cholesky decomposition of covariance """
                        """matrix failed. Your kernel may not be positive """
                        """definite. Scipy complained: %s""" % detail )
            print 'THe choleski matrix is: ', self._L
            print 'First solve (solve(self._L,self.y)) is: ', solve(self._L,self.y)
            self._alpha = solve(self._L.T, solve(self._L,self.y))
            print 'The second_solve(_alpha)->(solve(self._L.T, solve(self._L,self.y))) value is: ', self._alpha
            self.LL = (
                    - self.n * math.log( 2.0 * math.pi )
                    - ( self.y.T * self._alpha )[0,0]
            ) / 2.0
        print 'The self.alpha value is: ', self._alpha
        print 'The self.LL valueis : ', self.LL
        #import IPython; IPython.Debugger.Pdb().set_trace()
        self.LL -= log(diagonal(self._L)).sum()
        print 'The self.LL after diagonal sum is valueis : ', self.LL
        # print self.LL
        # print 'Done updating'


    def predict( self, x_star ):
        """
        Predict the process's values on the input values
        @arg x_star: Prediction points
        @return: ( mean, variance, LL )
        where mean are the predicted means, variance are the predicted
        variances and LL is the log likelihood of the data for the given
        value of the parameters (i.e. not integrating over hyperparameters)
        """
        print ''
        print 'In the Predict function .........................................'
        from numpy.linalg import solve
        import types
        # print 'Predicting'
        if 0 == len(self.X):
        	print 'I am in the if of predict'
        	f_star_mean = matrix(zeros((len(x_star),1), numpy.float64))
        	v = matrix(zeros((0,len(x_star)), numpy.float64))
        else:
        	print 'I am in the else of predict'
        	print 'self.X is: ', self.X
        	print 'x_star is: ', x_star
        	k_star = self.calc_covariance( self.X, x_star)
        	print 'k_star.shape is: ', k_star.shape
        	print 'shape of k_star.T is ', k_star.T.shape
        	print 'The shape of alpha is: ', self._alpha.shape
        	f_star_mean = k_star.T * self._alpha
        	if 0 == len(x_star): # no training data
        		v = matrix(zeros((0,len(x_star)), numpy.float64))
        	else:
        		print 'We have training Data (In our case the training data is the prior)'
        		print 'the self.L before the solve is: ', self._L
        		v = solve( self._L, k_star )
        		print 'The v after solving the Choleski decomposed symmetric matrix is: ', v
        cov = self.calc_covariance(x_star)
        print 'The covariance is: ', cov.shape, cov
        # V_f_star = cov - v.T * v
        V_f_star = cov - np.dot(v.T, v)
        # print 'The mean of the x_star is: ', f_star_mean.shape, f_star_mean
        # print 'The Variance of the x_star is: ', V_f_star.shape, V_f_star
        # print 'Done predicting'
        #import IPython; IPython.Debugger.Pdb().set_trace()
        return ( f_star_mean, V_f_star, self.LL )  # self.LL

    def calc_covariance( self, x1, x2 = None ):
        """
        Calculate covariance matrix of x1 against x2
        if x2 is None calculate symmetric matrix
        """
        # print 'THe x axis for covariance x1 is : ', x1
        # print 'THe y axis for covariance x2 is : ', x2
        print 'Calculating the covariance .........................'
        print 'x1 in covariance is: ', x1
        print 'x2 in covariance is: ', x2

        if any(x1):
        	if x2 is None:
        		sqdist = spdist.cdist(x1,x1,'sqeuclidean')
        	elif x2 is not None:
        		sqdist = spdist.cdist(x1,x2,'sqeuclidean')
        else:
        	return np.array([])
        return np.exp(-0.5 * sqdist)



def gp_plot_samples_from(gp, support, num_samples=1):
    """
    Plot samples from a Gaussian process.
    """
    print ''
    print 'In gp_plot_samples_from ...................'
    from pylab import plot
    mean, sigma, LL = gp.predict(support)
    print 'The mean is: ', mean.shape, mean
    print 'The variance is: ', sigma.shape, sigma
    print 'The value of LL is: ', LL
    gp_plot_prediction(support, mean, sigma)
    for i in xrange(num_samples):
        sample = numpy.random.multivariate_normal(
            asarray(mean).reshape(len(support),),
            sigma
        )
        print ''
        print 'The Sample is: ', sample
        plot([x[0] for x in support], sample)



def gp_plot_prediction(predict_x, mean, variance = None):
    """
    Plot a gp's prediction using pylab including error bars if variance specified
    Error bars are 2 * standard_deviation as in `Gaussian Processes for Machine Learning`__ by Rasmussen and Williams. 
    __ http://www.amazon.co.uk/Gaussian-Processes-Learning-Adaptive-Computation/dp/026218253X/
    """
    print ''
    print 'In the gp_plot_prediction function .....................................................' 
    from pylab import plot, concatenate, fill
    import matplotlib.pyplot as plt

    if None != variance:
    	print 'In the if condition of gp_plot_prediction'
        # check variances are just about +ve - could signify a bug if not
        assert diagonal( variance ).all() > -1e-10
        print 'mean_flatis :', mean.flat
        print 'diagonal variance is: ', diagonal(variance)	# all the diagonal elements in the covariance matrix is 1
        data = [(x,y,max(v,0.0)) for x,y,v in zip( predict_x, mean.flat, diagonal( variance ))]
        # print 'the data is', data
    else:
    	print 'In the else of gp_plot_prediction'
        data = [
                (x,y)
                for x,y
                in zip( predict_x, mean )
        ]
    data.sort( key = lambda d: d[0] ) # sort on X axis
    print 'data after sort is: ', data
    predict_x = [ d[0] for d in data ]
    predict_y = asarray( [ d[1] for d in data ] )
    print ''
    print 'The predict_x(xaxis) is: ', predict_x
    print 'The predict_y(yaxis) is: ', predict_y

    plt.plot( predict_x, predict_y, color='k', linestyle=':' )
    
    if None != variance:
    	print 'The variance is not None'
        sd = sqrt( asarray( [ d[2] for d in data ] ) )
        var_x = concatenate((predict_x, predict_x[::-1]))
        var_y = concatenate((predict_y + 2.0 * sd, (predict_y - 2.0 * sd)[::-1]))
        p = fill(var_x, var_y, edgecolor='w', facecolor='#d3d3d3')
        plt.plot( predict_x, predict_y + 2.0 * sd, 'k--' )
        plt.plot( predict_x, predict_y - 2.0 * sd, 'k--' )    
    plt.show('hold')



# import types
print '#################################################   FOR THE PRIOR    ################################################'
support = gp_1D_X_range(-10.0, 10.01, .125)
figure()
# support = [[1],[3],[6]]
# print support
gp = GaussianProcess([], [])
gp_plot_samples_from(gp, support, num_samples=3)
xlabel('x')
ylabel('f(x)')
title('Samples from the prior')
savefig('/home/sardendhu/Desktop/samples_from_prior.png')
savefig('/home/sardendhu/Desktop/samples_from_prior.eps')


print ''
print ''
print ''
print ''
print '#################################################   FOR THE POSTERIOR    ############################################'

X = [[-5.], [-2.], [3.], [3.5]]
Y = [2.5, 2, -.5, 0.]
X = [[1],[3],[6]]
Y = [2,3,7]
# # #
# # # Plot samples from posterior
# # #
figure()
plot([x[0] for x in X], Y, 'ks')
gp = GaussianProcess(X, Y)
gp_plot_samples_from(gp, support, num_samples=3)
# # xlabel('x')
# # ylabel('f(x)')
# # title('Samples from the posterior')
# # savefig('samples_from_posterior.png')
# # savefig('samples_from_posterior.eps')