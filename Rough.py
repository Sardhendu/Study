
a = [4, 5.09, 8.641, 11.646, 13.557, 8.753, 8.942, 14.815, 11.179, 8.168]
b = [8.641, 11.646, 13.557, 8.753, 8.942, 14.815, 11.179, 8.168]
c =[4, 5.09, 8.641, 11.646, 13.557, 8.753, 8.942, 14.815]

A = [235,239,244.090,252.731,264.377,277.934,286.687,295.629,310.444,325.112]
B = [239,244.090,252.731,264.377,277.934,286.687,295.629,310.444,325.112,336.291]

A1 = [235,239,244.090,252.731,264.377,277.934,286.687,295.629]
B1 = [244.090,252.731,264.377,277.934,286.687,295.629,310.444,325.112]



mean_zt = sum([j for i,j in enumerate(zt) if i >=2])/(10-2+1)

zt = [4,5.09,8.641,11.646,13.557,8.753,8.942,14.815,14.668,11.179]
zt1 = [4,5.09,8.641,11.646,13.557,8.753,8.942,14.815]
zt2 = [8.641,11.646,13.557,8.753,8.942,14.815,14.668,11.179]


zt = [4,5.09,8.641,11.646,13.557,8.753,8.942,14.815,14.668]
zt1 = [4,5.09,8.641,11.646,13.557,8.753,8.942,14.815]
zt2 = [8.641,11.646,13.557,8.753,8.942,14.815,14.668,11.179]



#### Finale
A = [235,239,244.090,252.731,264.377,277.934,286.687,295.629,310.444,325.112]
B = [239,244.090,252.731,264.377,277.934,286.687,295.629,310.444,325.112,336.291]
zt_zt1 = [b-a for a,b in zip(A,B)]

mean_zt = sum(zt_zt1)/(10-2+1)

sum([(zt-mean_zt)*(zt_2-mean_zt) for zt, zt_2 in zip(zt_zt1[0:8],zt_zt1[2:10])])

sum([(zt-mean_zt)*(zt-mean_zt) for zt in zt_zt1])



######### Data Mining
alpha = 0.01
beta = 0.5

alpha = 0.0020548
beta = 0.1875
# mean = 500+320+640/3 = 1460
# mean = 
A = [3.06, 500*alpha, 6*beta]  
B = [2.68, 320*alpha, 4*beta]
C = [2.92, 640*alpha, 6*beta]

np.dot(A,B)/(pow(sum([pow(i,2) for i in A]), 0.5) * pow(sum([pow(i,2) for i in B]), 0.5))
np.dot(A,C)/(pow(sum([pow(i,2) for i in A]), 0.5) * pow(sum([pow(i,2) for i in C]), 0.5))
np.dot(B,C)/(pow(sum([pow(i,2) for i in B]), 0.5) * pow(sum([pow(i,2) for i in C]), 0.5))

A = [0.67,1.67,0,1.67,-2.33,0,-0.33,-1.33]
B = [0,0.67,1.67,0.67,-1.33,-0.33,-1.33,0]
C = [-1,0,-2,0,0,1,2,0]

np.dot(A,B)/(pow(sum([pow(i,2) for i in A]), 0.5) * pow(sum([pow(i,2) for i in B]), 0.5))

np.dot(A,C)/(pow(sum([pow(i,2) for i in A]), 0.5) * pow(sum([pow(i,2) for i in C]), 0.5))

np.dot(B,C)/(pow(sum([pow(i,2) for i in B]), 0.5) * pow(sum([pow(i,2) for i in C]), 0.5))



########## On information gain
# Information Gain = Entropy(Class) - weighted[Entropy(Class|split(x))]

from __future__ import division
import numpy as np


a = 4
b = 5
c = a + b 
entropy_before = -1 * ((a/c)*np.log2(a/c) + (b/c)*np.log2(b/c))


tot = 9 
d=3
e=1
f=d+e
weight_x1 = f/tot
g=4
h=1
i=g+h
weight_x2 = i/tot

entropy_x1 = weight_x1 * ( (-1 * (d/f)*np.log2(d/f)) + (-1 * (e/f)*np.log2(e/f)) )
entropy_x2 = weight_x2 * (-1 * ((g/i)*np.log2(g/i) + (h/i)*np.log2(h/i)))

entropy_after = entropy_x1 + entropy_x2

Information_gain = entropy_before - entropy_after



# For three class category
a = 3
b = 3
c = 3
d = a + b + c
entropy_before = -1 * ((a/d)*np.log2(a/d) + (b/d)*np.log2(b/d) + (c/d)*np.log2(c/d))



