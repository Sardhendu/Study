import numpy as np

class Mean(object):
	'''
		Base class for Mean
	'''
	def __init__():
		note =  'Inside the Base class of the Mean'
		print note

	def getMean():
		'''
			Fetch the Mean vector based on the given Input
		'''
		print 'MEAN parent'
		pass

class ZeroMean(Mean):
    '''
    	Zero mean.
    '''

    def __init__(self):
        note = 'Inside the __init__ function ZeroMean'
        print note

    def getMean(self, a=None):
        n, D = a.shape
        A = np.zeros((n,1))
        return A

    def getDerMatrix(self, x=None, der=None):
        n, D = x.shape
        A = np.zeros((n,1))
        return A
