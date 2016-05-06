import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np




pos_def_mat = np.array([[2,6],[6,20]], dtype = int)
pos_sem_def_mat = np.array([[2,6],[6,18]], dtype = int)
non_pos_def_mat = np.array([[2,6],[6,7]], dtype = int)

# x_vect = np.array([[x1],[x2]], dtype = float)

print pos_def_mat
print pos_sem_def_mat
print non_pos_def_mat
# pos_def_func = 2*pow(x1,2) + 12*x1*x2 + 20*pow(x2)
# pos_sem_def_funct = 

X1 = np.linspace(-5,5,20)
X2 = np.linspace(-5,5,20)

x1_vect = []
x2_vect = []
xtAx_pos_def = []
xtAx_pos_sem_def = []
xtAx_non_pos_def = []
for x1 in X1:
	for x2 in X2:
		x1_vect.append(x1) 
		x2_vect.append(x2)
		x_vect = np.array([[x1],[x2]], dtype = float)
		xtAx_pos_def.append(np.dot(x_vect.T, np.dot(pos_def_mat, x_vect))[0][0])
		xtAx_pos_sem_def.append(np.dot(x_vect.T, np.dot(pos_sem_def_mat, x_vect))[0][0])
		xtAx_non_pos_def.append(np.dot(x_vect.T, np.dot(non_pos_def_mat, x_vect))[0][0])
		# print xtAx_pos_def	

# for i in range(0,50):
# 	x_vect = np.array([[X1[i]],[X2[i]]], dtype = float)
# 	xtAx_pos_def.append(np.dot(x_vect.T, np.dot(pos_def_mat, x_vect))[0][0])
# 	xtAx_pos_sem_def.append(np.dot(x_vect.T, np.dot(pos_sem_def_mat, x_vect))[0][0])
# 	xtAx_non_pos_def.append(np.dot(x_vect.T, np.dot(non_pos_def_mat, x_vect))[0][0])
		# print xtAx_pos_def


print np.min(xtAx_pos_def), np.max(xtAx_pos_def)
print ''
print np.min(xtAx_pos_sem_def), np.max(xtAx_pos_sem_def)
print ''
print np.min(xtAx_non_pos_def), np.max(xtAx_non_pos_def)

fig = plt.figure()
ax = fig.add_subplot(311, projection='3d')
ax.scatter(x1_vect, x2_vect, xtAx_pos_def, c='r', marker='o')
ax = fig.add_subplot(312, projection='3d')
ax.scatter(x1_vect, x2_vect, xtAx_pos_sem_def, c='r', marker='o')
ax = fig.add_subplot(313, projection='3d')
ax.scatter(x1_vect, x2_vect, xtAx_non_pos_def, c='r', marker='o')
plt.show('hold') 