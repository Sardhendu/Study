from __future__ import division
import numpy as np

import matplotlib
import pylab
import matplotlib.pyplot as plt
import math



def plot_GaussianKernel (horizontal_axis, vertical_axis, color):
	matplotlib.use('TkAgg')
	plt.figure(1)
	plt.ion()
	plt.plot(horizontal_axis, vertical_axis, color = color)
	plt.axis([0, 6, 0, 1])
    



X = np.array([1,1.2,3.2,4,5.1], dtype = 'float')
Y = np.array([23,17,12,27,8], dtype = 'float')
var = 1.74
color = ['r', 'b', 'g', 'r', 'b']
X_kernel_matrix = []
for no,i in enumerate(X):
	array = [np.exp(-pow(j-i,2)/(2*pow(var,2))) for j in np.arange(0,6,0.1)]    # We say that the point i is sampled from gaussian with mean i
	X_kernel_matrix.append(array)
	plot_GaussianKernel(np.arange(0,6,0.1), array, color[no])

plt.show('hold')

X_kernel_matrix = np.array(X_kernel_matrix)
xTx = np.dot(X_kernel_matrix.T, X_kernel_matrix)
xTx_inv = np.linalg.inv(xTx)

print xTx_inv.shape
# print np.dot(X_kernel_matrix.T[0], np.array(Y).T)
#np.exp(-pow(0-i)/(2*pow(0.5,2)))




# import matplotlib.pyplot as plt
# plt.plot([1,2,3,4], [1,4,9,16], 'ro')
# plt.axis([0, 6, 0, 20])
# plt.show()