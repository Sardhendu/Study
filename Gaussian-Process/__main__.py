from GaussianProcess import GPR
from sklearn import preprocessing
import Tools
import numpy as np


###########   Explanation   ##############
"""
	x and y are the training set and x*(xstar) is the test set.
	First we calculate the covariance matrix (Kernel) of the x

		__    __       __
	1: |f |= |k 	 k*	 |
	   |f*|	 |k*.T   k** |
	   |__|  |___	   __|
		   F is basically the joint, In the above kernel matrix, we are given k (cov(x,x)) from the training data. We compute kstar and kstar_star  
	2: we then deconpose using Choleski decomposition and find L (The choleski decomposition matrix)
	3. We solve and find alpha () using the L and y
	4. We use the formula to find conditional distribution for f* given x*, x, f
		p(f*|x*,x,f) = N~(f*|mean*,sigma*)    where mean* is the ystar and sigma* is the ystar_var
		where,
			mean* = mean(x*) + 

"""

x = [1,3,4.4,6,7,8,9,11]
y = [2,3,4.8,7,7.3,8,9.2,10]

xstar = Tools.gp_1D_X_range(-10.0, 10.01, .125)
xstar  = np.linspace(-11,11,50)

x_scaled = preprocessing.scale(x)
y_scaled = preprocessing.scale(y)
xstar_scaled = preprocessing.scale(xstar)

model = GPR()
print model.meanFunc                    # default prior mean
print model.covFunc                     # default prior covariance
print model.infFunc                     # inference method
model.getPrior(no_of_samples=100)
model.plot_samples(model.x_prior,model.prior_fnc,s=100)
model.getPosterior(x, y)
model.predict(xstar)
data = [(x1,y1,max(v1,0.0)) for x1,y1,v1 in zip(xstar, model.ystar, model.ystar_var)]
print data
model.plot(x, y, xstar)		