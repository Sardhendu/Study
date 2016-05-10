




import numpy as np
import matplotlib.pyplot as plt
import sys
from math import sqrt
import scipy.linalg.lapack as lapack

def jitchol(A,maxtries=5):
    ''' Copyright (c) 2012, GPy authors (James Hensman, Nicolo Fusi, Ricardo Andrade,
        Nicolas Durrande, Alan Saul, Max Zwiessele, Neil D. Lawrence).
    All rights reserved
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of the <organization> nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    :param A: the matrixed to be decomposited
    :param int maxtries: number of iterations of adding jitters
    '''    
    A = np.asfortranarray(A)
    print  'tools: The new revised Covariance matrix A is: ', A 
    L, info = lapack.dpotrf(A, lower=1)
    print 'tools: L and info is: ', L, info
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        print 'tools: The Diagonal Matrix of A is: ', diagA
        if np.any(diagA <= 0.):
            raise np.linalg.LinAlgError("kernel matrix not positive definite: non-positive diagonal elements")
        jitter = diagA.mean() * 1e-9
        while maxtries > 0 and np.isfinite(jitter):
            print('Warning: adding jitter of {:.10e} to diagnol of kernel matrix for numerical stability'.format(jitter))
            try:
                return np.linalg.cholesky(A + np.eye(A.shape[0]).T * jitter, lower=True)
            except:
                jitter *= 10
            finally:
                maxtries -= 1
        raise np.linalg.LinAlgError("kernel matrix not positive definite, even with jitter.")



def solve_chol(L, B):
    '''
    Solve linear equations from the Cholesky factorization.
    Solve A*X = B for X, where A is square, symmetric, positive definite. The
    input to the function is L the Cholesky decomposition of A and the matrix B.
    Example: X = solve_chol(chol(A),B)

    :param L: low trigular matrix (cholesky decomposition of A)
    :param B: matrix have the same first dimension of L
    :return: X = A \ B
    '''
    try:
        assert(L.shape[0] == L.shape[1] and L.shape[0] == B.shape[0])
    except AssertionError:
        raise Exception('Wrong sizes of matrix arguments in solve_chol.py');
    X = np.linalg.solve(L,np.linalg.solve(L.T,B))
    return X