import numpy as np
from scipy import linalg



def sort_eigen_velues_vector(eigval, eigvect):
	idx = eigval.argsort()[::-1]   
	eigval = eigval[idx]
	eigvect = eigvect[:,idx]
	return eigval, eigvect

def chk_if_pos_def(mat_in):
	eigval_, eig_vect = np.linalg.eig(mat_in)
	det_chk = np.linalg.det(mat_in)
	print 'For a matrix (A) to be Posive definite, its eigen values should be +ve, its determinant should be +ve and the eq. X.T(A)X should be +ve'
	print 'eigen values ', eigval_
	print 'determinant ' , det_chk
	print 'sum of eigen values ', np.sum(eigval_)
	print 'product of eigen values', np.prod(eigval_)
	print ''
	return eigval_, eig_vect

def formula_1(lamda_diag, S, Sinv):		# S(lamda_diag)Sinv
	'''Formula: A = S(lamda_diag)Sinv'''
	S_ld_Sinv = np.dot(S, np.dot(lamda_diag, Sinv))
	print 'S_ld_Sinv = A :', S_ld_Sinv
	print ''
	return S_ld_Sinv

def formula_2(A, S, Sinv):				# Sinv(A)S
	'''Formula: Sinv(A)S = lamda_diag'''
	Sinv_A_S = np.dot(Sinv, np.dot(A, S))
	return Sinv_A_S

def formula_3(A, M, Minv):				# Minv(A)M
	'''Formula: Minv(A)M = B'''
	Minv_A_M = np.dot(Minv, np.dot(A, M))
	return Minv_A_M	

def chk_if_similar_using_eigval(A, B):	# Eigen values of the matrices should match
	'''If similar then the eigen values of both the matrices will be same'''
	eigval_A, eigvect_A = np.linalg.eig(A)
	eigval_B, eigvect_B = np.linalg.eig(B)
	eigval_A, eigvect_A = sort_eigen_velues_vector(eigval_A, eigvect_A)
	eigval_B, eigvect_B = sort_eigen_velues_vector(eigval_B, eigvect_B)
	return eigval_A, eigval_B
	# if eigval_1 == eigval_2 == True:
	# 	print 'Yes the matrices are similar: ', matrix1, matrix2

def chk_if_similar_using_eigvect(A, B, Minv):
	'''If Minv(A)M = B then A and B are similar, hence [eig_vect of B = Minv(eig_vect of A)]'''
	eigval_A, eigvect_A = np.linalg.eig(A)
	eigval_B, eigvect_B = np.linalg.eig(B)
	eigval_A, eigvect_A = sort_eigen_velues_vector(eigval_A, eigvect_A)
	eigval_B, eigvect_B = sort_eigen_velues_vector(eigval_B, eigvect_B)
	eigvect_B_nw = np.dot(Minv,eigvect_A)
	return eigvect_A.round(decimals=0), eigvect_B.round(decimals=0), eigvect_B_nw.round(decimals=0)







###############################################################################################################################
print '.............................. Good case, when all the eigen values are different .....................................'
###############################################################################################################################

'''
	1. Matrices that have the same Eigen values belongs to the same family, however, they can have different eigen vectors.
	2. If  Sinv(A)S = lamda_diag   and   Minv(A)M = B,  then A, B and lamda_diag are similar matrices
	3. The sum of eigen values should be equal to the trace of the matrix (sum of diagonal elements)
	4. The product of eigen values should be equal to the determinant of the matrix.
	5. If A and B are similar matrices lets say, Minv(A)M = B then the [eig_vector of B = Minv(eig_vector of A)  
	   Example:
		Using (2), if Sinv(A)S = lamda_diag, which is in the form AX = lamda(X), here X is eig_vector so S is eig_vector of A, therefore [eig_vector of B = Minv(S)]
'''

rect_m1 = X = np.array([[2,4],[6,7], [8,11]], dtype = int)  	# Is a rectangular martix
print 'X = ', X
print ''
sq_sm_m1 = A = np.dot(X.T, X)#np.array([[2,1],[1,2]], dtype = int)# Is a square symmetric matrix
print 'A = (X.T)X = ', A
print ''
eigval_, eig_vect = chk_if_pos_def(A)


print '---------------------------------------------------------------------------------------------------------------------'
S = eig_vect
print 'S = eig(A) = eigvect_A = ', S
print ''
Sinv = S_trans = np.linalg.inv(eig_vect)					# Since eigen vector matrices are square symmetric and 
print 'Sinv = Strans = ', Sinv
print ''
Sinv_A_S = lamda_diag = formula_2(A, S, Sinv)
print 'Sinv_A_S = lamda_diag:', Sinv_A_S
print ''

print '---------------------------------------------------------------------------------------------------------------------'
M = np.array([[1, 4], [0, 1]])
# M = np.array([[5, 2], [1, 6]])								# We cook an arbitarary matrix M )
print 'M = ', M
print ''
Minv = np.linalg.inv(M)
print 'Minv = ', Minv
print ''
Minv_A_M = B = formula_3(A, M, Minv)
print 'Minv_A_M = B:', Minv_A_M
print ''

print '---------------------------------------------------------------------------------------------------------------------'
eigval_ld,eigval_A = chk_if_similar_using_eigval(lamda_diag, A)
print 'eig(lamda_diag) = eig(Sinv_A_S) = eigval_ld = ', eigval_ld
print 'eig(A) = eigval_A =: ', eigval_A
print ''
eigval_A,eigval_B = chk_if_similar_using_eigval(A, B)
print 'eig(A) = eigval_A = ', eigval_A
print 'eig(B) = eigval_B =: ', eigval_B
print ''
eigval_ld,eigval_B = chk_if_similar_using_eigval(lamda_diag, B)
print 'eig(lamda_diag) = eig(Sinv_A_S) = eigval_ld = ', eigval_ld
print 'eig(B) = eigval_B =: ', eigval_B
print ''

print '---------------------------------------------------------------------------------------------------------------------'
eigvect_A, eigvect_ld, eigvect_ld_nw = chk_if_similar_using_eigvect(A, lamda_diag, Sinv)
print 'S = eig(A) = eigvect_A = ', eigvect_A
print 'T = eig(lamda_diag) = eigvect_ld = ', eigvect_ld
print 'T = (Sinv(S)) = eigvect_ld = ', eigvect_ld_nw
print ''
eigvect_A, eigvect_B, eigvect_B_nw = chk_if_similar_using_eigvect(A, B, Minv)
print 'S = eig(A) = eigvect_A = ', eigvect_A
print 'T = eig(B) = eigvect_B = ', eigvect_B
print 'T = (Minv(S)) = eigvect_B = ', eigvect_B_nw
print ''
print ''
print ''
print ''




###############################################################################################################################
print '........................  Bad case, when one or multiple eigen values are same  .......................................'
###############################################################################################################################

'''
 	When we have two or more eigen values same, we may not get all sets of Eigen vectors. This is a challange we get in finding family of matrices. Lets take an example below.
 	A = [[4,0]      eigval1 = 4
 		 [0,4]]		eigval2 = 4

 	C = [[4,1]      eigval1 = 4
 		 [0,4]]		eigval2 = 4

 	Both A and C has same eigen value but they are not similar (doesn't belong to the same family)
 	Lets see how
'''

A = np.array([[4,0],[0,4]], dtype=float)
print 'A = ', A
print ''
M = np.array([[5,1],[1,2]], dtype=float)		# Arbitarary
print 'M = ', M
print ''
Minv = np.linalg.inv(M)
print 'Minv = ', Minv
print '' 
Minv_A_M = B = formula_3(A, M, Minv)
print 'Minv_A_M = B = ', Minv_A_M
print ''
print 'B = A = Minv_A_B, which means the matrix A is the only one memeber in the family. Because our similarity theoram said if Minv_A_M = B then matrices A and B are similar. But here we get same output for A and B. For any matrix M the result Minv_A_M will always be = A'
print ''

print '---------------------------------------------------------------------------------------------------------------------'
C = np.array([[4,1],[0,4]], dtype=float)
Minv_C_M = D  = formula_3(C, M, Minv)
print 'Minv_C_M = D = ', Minv_C_M
print ''
print "C != D = Minv_C_B, which means the matrix C and D belongs to the smae family. Because our similarity theoram said if Minv_C_M = D then matrices C and D are similar and here we get different output for C and D. For different combinations of matrix M the result Minv_C_M will always be produce a different matrix D. and all D's will belong to the family of C"
print ''

print '---------------------------------------------------------------------------------------------------------------------'
eigval_C,eigval_D = chk_if_similar_using_eigval(C, D)
print 'eig(C) = eigval_C = ', eigval_C
print 'eig(D) = eigval_D =: ', eigval_D
print ''


'''
	What did we learn?
	Normally both out matrix A and C should be in the same family however, the family bifurcates. One family consists of only A and the other family consist of all other matrixes like C, D that have eigen value (4,4)
'''