import numpy as np

def rotate_matrix_anticlock(matrix):
    'Rotate anti clock wise'
    for i in np.arange(len(matrix)//2):
        rng_2 = len(matrix) -i - 1
        for j in np.arange(i, rng_2):
            # Here we find all teh components to be exchanges
            tup = (i, j)
            first = matrix[i, j]

            for k in np.arange(3):   # THIS 3 value is constant
                i_ = matrix.shape[0]-1 - tup[1]
                j_ = tup[0]
                sec = matrix[i_, j_]
                matrix[i_,j_] = first
                first = sec
                tup = (i_, j_)

            matrix[i,j] = first
    return matrix

def rotate_matrix_clock(matrix):
    'Rotate anti clock wise'
    for i in np.arange(len(matrix)//2):
        rng_2 = len(matrix) -i - 1
        for j in np.arange(i, rng_2):
            # Here we find all teh components to be exchanges
            tup = (i, j)
            first = matrix[i, j]

            for k in np.arange(3):   # THIS 3 value is constant
                i_ = tup[1]
                j_ = matrix.shape[0]-1 - tup[0]
                sec = matrix[i_, j_]
                matrix[i_,j_] = first
                first = sec
                tup = (i_, j_)

            matrix[i,j] = first
    return matrix


rotmat = rotate_matrix_clock(np.array([[1,2,3,4],
                                [5,6,7,8],
                                [9,10,11,12],
                                [13,14,15,16]]))
#
# rotmat = rotate_matrix_anticlock(np.array([[1,2,3,4,17],
#                                 [5,6,7,8,18],
#                                 [9,10,11,12,19],
#                                 [13,14,15,16,20],
#                                  [1,2,1,3,5]]))

print(rotmat)