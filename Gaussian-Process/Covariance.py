from __future__ import division
import numpy as np
import scipy.spatial.distance as spdist




class Covariance(object):
	'''
		Base class of Covariance
	'''
	def __init__(self, hyp = [1,1]):
		note = 'Inside the __init__ of CovKernel'
		self.l = hyp[0]					# Controls the variance of the distribution, the larger th l and smaller the variance, or we can also say that l controls the smoothness of the function.
		self.sigma = hyp[1]		
		print note


class SqExp(Covariance):
	'''
		This covariance form (sqExp) is the squared exponential Kernel

			|Formula:  		exp(-0.5 * pow((x-y), 2)), where x and y are data points
			|Input: 		Vector a and Vector b
			|Output:		Covariance matrix 
							[[a0b0   a0b1   ......   a0bn]
					 		 [a1b0   a1b1   ......   a1bn]
					 			.
					 			.
					 		 [anb0   anb1   ......   anbn]]

		All three Function sq_exp_vect, sq_exp_vect_scipy and sq_exp_loop should give the same output for the input a,b. 
	'''
	def __init__(self):
		super(SqExp, self).__init__()
		note = 'Inside the __init__ of SqExp'
		print note

	def sq_exp_vect(self, a, b):
		asquare = np.sum(pow(a,2),1).reshape(-1,1)    # The 1 in the sum is given, so that it doesnt add the whole matrix we just want to square the data point and organize them in matrix, try this : print np.sum([[2],[3]], 1) and print np.sum([[2],[3]])
		bsquare = np.sum(pow(b,2),1)
		twoab = 2*np.dot(a,b.T)
		sqdist = asquare + bsquare - twoab
		# print sqdist
		return np.exp(-0.5 * sqdist)	# We take lamda value to be 0.5

	def sq_exp_loop(self, a, b, n):
		a = a.reshape(1,-1)[0]					# We reshape to do the calculation in loop form
		b = b.reshape(1,-1)[0]
		kernel_matrix = np.zeros((n,n))
		for i, ival in enumerate(a):
			for j, jval in enumerate(b):
				# print np.exp(-0.5 * pow((ival - jval), 2))
				kernel_matrix[i][j] = np.exp(-0.5 * pow((ival - jval), 2))
		return kernel_matrix

	def sq_exp_vect_scipy(self, a=None, b=None, mode=None):
		if mode == 'test':
			nn,D = b.shape
			sqdist = np.zeros((nn,1))
		elif mode == 'self':
			sqdist = spdist.cdist(a,a,'sqeuclidean')
		elif mode == 'cross':		
			sqdist = spdist.cdist(a,b,'sqeuclidean')

		return pow(self.sigma,2) * np.exp((-0.5/self.l) * sqdist)	# We take lamda value to be 0.5


