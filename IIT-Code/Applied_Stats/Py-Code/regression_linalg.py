"""
	Module Creater: Sardhendu Mishra
	Details: Linear Regression using simple concepts from Linear Algebra
# 
"""


from __future__ import division

import numpy as np
import matplotlib

matplotlib.use('TkAgg')    # Since the matplotlib backend doesn't know we have to manually implement it 

from matplotlib import pyplot as plt


######################  Data Set 1   ##########################
# Input your data here:
# x = [1,2,3]
# y = [1,2,2]

# # Formula to Use = y=mx+c where y is the dependent or output dataset, x is the independent or input dataset, m is the slope and c is the y-intercept

# # By replacing y and x to the equation we get
# '''
# 	1.  y = m + c
# 	2.  2y =  2m + c
# 	3.  2y = 3m + c
# '''

# # System of Equations become: (Ax = B)

# A = np.matrix([ [ 1,1], [ 2,1] , [ 3, 1 ] ])
# X = np.matrix([ ["m"], ["c"] ])
# B = np.matrix([ [1], [2], [2] ])
######################  Data Set 1   ##########################


######################  Data Set 2   ##########################
GPA = [3.26, 2.60, 3.35, 2.86, 3.82, 2.21, 3.47]
Salary = [33.8, 29.8, 33.5, 30.4, 36.4, 27.6, 35.3]

x = GPA
y = Salary
# Formula to Use = y=mx+c where y is the dependent or output dataset, x is the independent or input dataset, m is the slope and c is the y-intercept

# By replacing y and x to the equation we get
'''
	1.  y = m + c
	2.  2y =  2m + c
	3.  2y = 3m + c
'''

# System of Equations become: (Ax = B)

A = np.matrix([ [3.26,1], [2.60,1] , [3.35, 1], [2.86, 1], [3.82, 1], [2.21, 1], [3.47, 1]])
X = np.matrix([ ["m"], ["c"] ])
B = np.matrix([ [33.8], [29.8], [33.5], [30.4], [36.4], [27.6], [35.3]])
######################  Data Set 2   ##########################


# Since A is not a square matrix (it is a rectangular matrix) and the colums are independent, we cannot solve the systems of equation 
# Therefore we use the formula     transpose(A).Ax = transpose(A).B

A_transpose = np.transpose(A)

A_transpose_A =  np.dot(A_transpose, A)
A_transpose_B =  np.dot(A_transpose, B)

print 'A is: ', A 
print 'X is: ', X
print 'B is: ', B
print 'A_transpose is: ', A_transpose
print 'A_transpose_A is: ',A_transpose_A
print 'A_transpose_B is: ',A_transpose_B
print ''
print 'The shape of A_transpose_A is: ', np.shape(A_transpose_A)
print 'The shape of X is: ', np.shape(X)
print 'The shape of B is: ', np.shape(B)
print 'The shape of A_transpose_B is: ', np.shape(A_transpose_B)

# The equation np.dot(A_transpose_A, X) becomes  

m , c  = np.linalg.solve(A_transpose_A, A_transpose_B)
m = float(np.array(m)[0][0])
c = float(np.array(c)[0][0])
print 'The slope is: ', m
print 'The y-intercept is: ', c


# Finding points of prediction whern given (m) the Slope and (c) the y-intercept
# point_1 = m * (x[0]) + c
expected_y =  [m*i+c for i in x]
print expected_y


# Plotting 
print max(x)
plt.ion()
plt.figure(1)
plt.subplot(211)
plt.plot(x,y,"ro")
plt.axis([0, max(x)+2, 0, max(y)+2])

plt.subplot(212)
plt.plot(x,y,"ro")
plt.plot(x, expected_y, "-")
plt.axis([0, max(x)+2, 0, max(y)+2])

plt.show("wait")
