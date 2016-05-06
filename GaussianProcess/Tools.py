import numpy as np


def gp_1D_X_range(xmin, xmax, step=1.):
    """
    @return: An array of points between xmin and xmax with the given step size. The shape of the array
    will be (N, 1) where N is the number of points as this is the format the GP code expects.
    Implementation uses numpy.arange so last element of array may be > xmax.
    """
    result = np.arange(xmin, xmax, step)
    result.resize((len(result), 1))
    return result