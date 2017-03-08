
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


a = 7
b = 4
c = a + b 
entropy_before = -1 * ((a/c)*np.log2(a/c) + (b/c)*np.log2(b/c))

gini_before = (a/c)

tot = 9 
d=3
e=1
f=d+e
weight_x1 = f/tot
g=5
h=2
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


2,2,2,1,2,2

tot = 11 
d=2
e=1
f=d+e
weight_x1 = f/tot
entropy_x2 = weight_x1 * (-1 * ((d/f)*np.log2(d/f) + (e/f)*np.log2(e/f)))
0.36363636363636365 + 


0.994 - ((2*0.36363636363636365)+ 0.25044431837849712)

g=5
h=2
i=g+h
weight_x2 = i/tot
entropy_x2 = weight_x2 * (-1 * ((g/i)*np.log2(g/i) + (h/i)*np.log2(h/i)))


(80+60)/(80+60+40+20)

p = 0.667
r = 0.8
(2*p*r)/(p+r)



((4/11) * (2/4) *(2/4)) + ((3/11)*(2/3)*(1/3)) + ((4/11)*(2/4)*(2/4))

0.09090909090909091+ 0.060606060606060594 + 0.09090909090909091



#################      1      #######
from __future__ import division
import numpy as np
n = 5
x = np.array([5,10,15,20,30])
y = np.array([0.15,0.25,0.30,0.50,0.75])


x_mean = np.mean(x)
y_mean = np.mean(y)

N = np.sum([(xi-x_mean)*(yi-y_mean) for xi,yi in zip(x,y)])
D = np.sum([pow((xi-x_mean), 2) for xi in x])

b1 = N/D
b0 = y_mean - (b1*x_mean)

y_hat = b0 + b1*x
SSE = (1/2)* np.sum([pow((yi-y_hat_i),2) for yi, y_hat_i in zip (y, y_hat)])

0.0089999999999999941

SSE = 15.300000000000001

##############      2      ###########




X	True labels f(X)	g1(X)	g2(X)	g3(X)
0	1	0	1	1
1	2	2	3	4
2	5	4	5	5
3	6	6	7	7
4	9	8	9	10


n = 5
m = 3

x = np.array([0,1,2,3,4])
y = np.array([1,2,5,6,9])

g1 = np.array([0,2,4,6,8])
g2 = np.array([1,3,5,7,9])
g3 = np.array([1,4,5,7,10])

g_bar_array = [(g1x+g2x+g3x) / m for g1x,g2x,g3x in zip(g1,g2,g3)]

[0.66666666666666663, 3.0, 4.666666666666667, 6.666666666666667, 9.0]

bias_sq = (1/5)*np.sum([pow((gx-fx),2) for gx, fx in zip (g_bar_array, y)])

bias = 0.33333333333333343

[for i,j in zip()]

(1/(3*5))*

value = []
for no,g_bar_i in enumerate(g_bar_array):
	aa = pow((g1[no] - g_bar_i), 2) 
	bb = pow((g2[no] - g_bar_i), 2) 
	cc = pow((g3[no] - g_bar_i), 2)
	value.append(aa+bb+cc)


mean_g1 = np.mean(g1)
mean_g2 = np.mean(g2)
mean_g3 = np.mean(g3)

bias = 







