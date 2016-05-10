from __future__ import print_function
import numpy as np

class Mean(object):
    '''
    The base function for mean function
    '''
    def __init__(self):
        super(Mean, self).__init__()
        self.hyp = []
        self.para = []


    def __repr__(self):
        strvalue =str(type(self))+': to get the mean vector or mean derviatives use: \n'+\
              'model.meanfunc.getMean()\n'+\
          'model.meanfunc.getDerMatrix()'
        return strvalue

    # overloading
    def __add__(self,mean):
        '''
        Overloading + operator.

        :param mean: mean function
        :return: an instance of SumOfMean
        '''
        return SumOfMean(self,mean)

    # overloading
    def __mul__(self,other):
        '''
        Overloading * operator.
        Using * for both multiplication with scalar and product of means
        depending on the type of the two objects.

        :param other: mean function as product or int/float as scalar
        :return: an instance of ScaleOfMean or ProductOfMean
        '''
        if isinstance(other, int) or isinstance(other, float):
            return ScaleOfMean(self,other)
        elif isinstance(other, Mean):
            return ProductOfMean(self,other)
        else:
            print("only numbers and Means are allowed for *")

    # overloading
    __rmul__ = __mul__

    # overloading
    def __pow__(self,number):
        '''
        Overloading ** operator.

        :param int number: power of the mean function
        :return: an instance of PowerOfMean
        '''
        if isinstance(number, int) and number > 0:
            return PowerOfMean(self,number)
        else:
            print("only non-zero integers are supported for **")

    def getMean(self, x=None):
        '''
        Get the mean vector based on the inputs.

        :param x: training data
        '''
        pass

    def getDerMatrix(self, x=None, der=None):
        '''
        Compute derivatives wrt. hyperparameters.

        :param x: training inputs
        :param int der: index of hyperparameter whose derivative to be computed

        :return: the corresponding derivative matrix
        '''
        pass


class Zero(Mean):
    '''Zero mean.'''
    def __init__(self):
        self.hyp = []       # Hyperparameter for mean
        self.name = '0'

    def getMean(self, x=None):
        n, D = x.shape
        A = np.zeros((n,1))
        return A

    def getDerMatrix(self, x=None, der=None):
        n, D = x.shape
        A = np.zeros((n,1))
        return A