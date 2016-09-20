from __future__ import division
import numpy as np


class Inference(object):
	def __init__(self):
		note = 'Inside the __init__ of Inference'

	def evaluation(self):
		'''
			| INPUT:
			| meanFunc: name of the mean function (see Mean.py)
			| covFunc: name of the covariance function (see Covariance.py)
			| likFunc: name of the likelihood function (see Liklihood.py)
			| x: (nxD) matrix of training inputs
			| y: 1d array (of size n) -- Labels

			| OUTPUT:
			| post(postStruct): struct representation of the (approximate) posterior containing:
			| nlZ: returned value of the negative log marginal likelihood
			| dnlZ(dnlZStruct): struct representation for derivatives of the negative log marginal likelihood
			| w.r.t. each hyperparameter.
			'''
		pass

	# def draw_samples():
	# 	pass


class Decompose(Inference):

	def __init__(self):
		note = 'Inside the __init__ of Decompose'
		
	def svd(self, K):
		'''
			Using vector form and Singular value decomposition.....................
			The svd ....
		'''
		U ,sigma, V = np.linalg.svd(K, full_matrices=True)
		sigma = np.diag(sigma)
		# f_prior = np.dot(np.dot(U, sigma), rand_sample)
		return np.dot(U, sigma)

	def cholesky(self, K, n):
		'''
			Using Vector form and Cholesky decomposition....................
			The Cholesky decomposition is of form (L L.T). And we have to draw random sample from the decomposition matrix to get the smooth function. Since directly drawing samples from the decomposition matrix is not kind of much feasible, we draw random samples with gaussian {N(0,I) (mean = 0 and variance is the Identity matrix)} and multiply it to our decomposition matrix (Covariance matrix) to make it a smooth prior
		'''
		# print 'cholesky, ', 1e-6*np.eye(n)
		L = np.linalg.cholesky(K + 1e-6*np.eye(n))
		# f_prior  = np.dot(L, np.random.normal(size = (n,n)))  # This prior is a smooth function over the dataset (xtest) as we decompose the kernel matrix which is square exponent covariance matix of the dataset xtest
		return L


	# def jitcholesky(self, K):
	# 	K = np.asfortranarray(K)



class GetSamples(Inference):
	def __init__(self):
		note = 'Inside the __init__ function of GetSample  ......'

	def draw_samples(self, L, s, n):	# L is the decomposition matrix, s is the no. of samples, and n is the no of data points
		'''
			We basically draw s samples from the decomposed matrix. Normally drawing samples directly from th decomposion matrix is difficult, so we draw sample from distribution N with mean =0 and L=the identity matrix i.e. N~(0,I) and then multiply them to the decomposition matrix, which is samples from f' = N~(0,I) * L
		'''
		
		f_prior = np.dot(L, (np.random.normal(size = (n,s))))

		return f_prior
