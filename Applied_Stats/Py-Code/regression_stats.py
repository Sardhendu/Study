"""
	Module Creater: Sardhendu Mishra
	Details: Linear Regression using concepts from statistics(with assuming anything from probability) Linear Algebra

"""

from __future__ import division
import numpy as np
import matplotlib

matplotlib.use('TkAgg')    # Since the matplotlib backend doesn't know we have to manually implement it 

from matplotlib import pyplot as plt
####	Formulas
#	b0 = ym - b1xm
#
#
#
###

# Data Set 1

def Salary_data():
	GPA = [3.26, 2.60, 3.35, 2.86, 3.82, 2.21, 3.47]
	Salary = [33.8, 29.8, 33.5, 30.4, 36.4, 27.6, 35.3]
	x = GPA
	y = Salary
	n = len(y)

	x_mean = np.mean(x)
	y_mean = np.mean(y)

	# Finding the y-intercept
	SSxy = np.sum([(xi-x_mean) * (yi-y_mean) for xi, yi in zip(x, y)])
	SSxx = np.sum([pow((xi-x_mean), 2) for xi in x])
	b1 = SSxy/SSxx


	# FInding the y-intercept
	b0 = y_mean - (b1*x_mean)

	# finding expected point for GPA (0-4)
	gpa_lin = np.linspace(0,4,num=100)

	# Getting the point estimation based on b0 and b1 for the given GPA's 
	Salary_Point_Estimation = [b0 + b1*xi for xi in x]

	# Total variation of the given y from the mean(y)- Total Variation in y
	SSyy = np.sum([pow((yi-y_mean),2) for yi in y])

	# Getting the Sum of Squared Error using the error fiunction (e) e = y - (b0 + b1(x)) where (y is the true value, e is the min squared error and x is the Batch_Size data point)
	SSE = np.sum([pow((y_true-y_est), 2) for y_est, y_true in zip(Salary_Point_Estimation, y)])
	MSE =SSE/(n-2)  # Also called as "variance" or even "s_square"

	# Getting the Sum of Square Regression using the estimated point and their variance fromt he mean (y)
	SSR = np.sum([pow((y_est-y_mean),2) for y_est in Salary_Point_Estimation])

	# Getting the total error or Total Variance (The Sum of Squared error + The Sum of Squared Regression error)
	SST = SSE + SSR 

	# Getting the Point Prediction based on the error or resideu 
	given_gpa = 3.25
	Salary_Point_Estimation_for_given_gpa = b0 + b1*given_gpa
	Salary_Point_Prediction_for_given_gpa = Salary_Point_Estimation_for_given_gpa + SSE

	print "The given GPA's are: ", GPA
	print "The respective Salarie's are: ", Salary
	print 'The Mean value of the (Independent variable) No_of_Copiers is: ', x_mean, pow(x_mean,2)
	print 'The Mean value of the (Dependent variable) No_of_Minutes is: ', y_mean
	print 'The y-intercept anf the slope are : ', b0, b1
	print 'The Point estimation for GPA 3.25 is: ', Salary_Point_Estimation_for_given_gpa

	print ''
	print 'Sum of Total variation of the obsoved x and y from their respective mean (SSxy): ', SSxy
	print 'Sum of Total variation of the obsoved x its respective mean (SSxx): ', SSxx, pow(SSxx,0.5)
	print 'Sum of Total variation of the obsoved y its respective mean (SSyy): ', SSyy
	print 'The Sum of Squared error (SSE) or the variance is: ', SSE
	print 'The Mean of Squared error (MSE) or the estimation of variance is: ', MSE, pow(MSE,0.5)
	print 'The Sum of Square Regression Error (SSR) is: ', SSR
	print 'The Sum of Square Total Error (SST): ', SST


	# Ploting Graph
	plt.ion()
	plt.figure(1)
	plt.subplot(211)
	plt.plot(GPA,Salary,"ro")
	plt.axis([0, max(x)+2, 0, max(y)+2])

	plt.subplot(212)
	plt.plot(x,y,"ro")
	plt.plot(GPA, Salary_Point_Estimation, "-")
	plt.axis([0, max(x)+2, 0, max(y)+2])

	plt.show("wait")


##################################################################################################################################################

# Data Copier Service Time
def copier_service():
	No_of_Copiers = [4,2,5,7,1,3,4,5,2,4,6]  # [-2,-1,1,4]
	No_of_Minutes = [109,58,138,189,37,82,103,134,68,112,154]  # [-3,-1,2,3]
	x = No_of_Copiers
	y = No_of_Minutes
	n = len(y)

	x_mean = np.mean(x)
	y_mean = np.mean(y)

	# Finding the y-intercept
	SSxy = np.sum([(xi-x_mean) * (yi-y_mean) for xi, yi in zip(x, y)])
	SSxx = np.sum([pow((xi-x_mean), 2) for xi in x])
	b1 = SSxy/SSxx

	# FInding the y-intercept
	b0 = y_mean - (b1*x_mean)

	# Getting the point estimation based on b0 and b1 for the given GPA's 
	No_of_Minutes_Point_Estimation = [b0 + b1*xi for xi in x]

	# Total variation of the given y from the mean(y)- Total Variation in y
	SSyy = np.sum([pow((yi-y_mean),2) for yi in y])

	# Getting the Sum of Squared Error using the error fiunction (e) e = y - (b0 + b1(x)) where (y is the true value, e is the min squared error and x is the No_of_Copiers data point). The Squared error term for each x also tells us about how much of the varition in y is not measured by the regression line.
	SSE = np.sum([pow((y_true-y_est), 2) for y_true, y_est in zip(y, No_of_Minutes_Point_Estimation)])
	MSE = SSE/(n-2)	# Getting the Mean Squared Error also can be said as the variance

	# Total variation of the estimated y from the mean(y) "Sum of Square Regression" 
	SSR = np.sum([pow((y_est-y_mean),2) for y_est in No_of_Minutes_Point_Estimation])

	# Getting the total error or Total Variation in y (The Sum of Squared error + The Sum of Squared Regression error)
	SST = SSE + SSR 

	'''
		####  Important note  ####
		The value of SSyy and SST are the same because:
		SST = SSE + SSR
			SSE = square distance of y_est and y_true 
			SSR = square distance of y_est from y_mean
			Since, 
	'''

	# Calculating the (R-Squared Error) or Coefficient of Determination (The percentage of total variation is described by the line or by the variation in x- The percentage of variance that can be Explained)
	# SSR is the variation described by the regression line (Eplained Variation), and
	# SSE is the variation not described by thr regression line (Unexplained variation)
	r_sq_error = SSR/SST
	Det_coef = 1-(SSE/SST)   # 1 - (% of variation unexplained) =  % of variation explained

	# Correlation coeficient between X and Y
	corr_coef = np.corrcoef(x, y)

	given_no_of_copiers = 4
	No_of_minutes_Point_Estimation_for_given_no_of_copiers = b0 + b1*given_no_of_copiers
	No_of_minutes_Point_Prediction_for_given_no_of_copiers = No_of_minutes_Point_Estimation_for_given_no_of_copiers + e


	print 'The given data point of the No_of_Copiers are: ', x
	print 'The respective data points of No_of_Minutes are: ', y
	print 'The Mean value of the (Independent variable) No_of_Copiers is: ', x_mean
	print 'The Mean value of the (Dependent variable) No_of_Minutes is: ', y_mean
	print 'The y-intercept and the slope are : ', b0, b1
	print 'The Point Estimation for 4 copiers is: ', No_of_minutes_Point_Estimation_for_given_no_of_copiers
	print 'The Point Prediction for 4 copiers is: ', No_of_minutes_Point_Prediction_for_given_no_of_copiers
	print ''
	print 'Sum of Total variation of the obsoved x and y from their respective mean (SSxy): ', SSxy
	print 'Sum of Total variation of the obsoved x its respective mean (SSxx): ', SSxx
	print 'Sum of Total variation of the obsoved y its respective mean (SSyy): ', SSyy
	print 'The Sum of Squared error (SSE) is: ', SSE
	print 'The Mean of Squared error (MSE) is: ', MSE
	print 'The Sum of Square Regression Error (SSR) is: ', SSR
	print 'The Sum of Square Total Error (SST): ', SST
	print 'The R-squared error and coefficient of Determination are: ', r_sq_error, Det_coef
	print 'The Estimated No_of_Minutes_Point_Estimation is: ', No_of_Minutes_Point_Estimation
	print 'The Cor-relation cofficient of the Batch_Size and Labor_Cost is: ', corr_coef

	# Ploting Graph
	plt.ion()
	plt.figure(1)
	plt.subplot(211)
	plt.plot(x,y,"ro")
	plt.axis([min(x)-2, max(x)+2, min(y)-2, max(y)+2])

	plt.subplot(212)
	plt.plot(x,y,"ro")
	plt.plot(x, No_of_Minutes_Point_Estimation, "-")
	plt.axhline(y=y_mean, color = 'r')
	plt.axis([min(x)-2, max(x)+2, min(y)-2, max(y)+2])

	plt.show("wait")




##################################################################################################################################################



def labor_cost_data():
	Batch_Size = [5,62,35,12,83,14,46,52,23,100,41,75]
	Labor_Cost = [71,663,381,138,861,145,493,548,251,1024,435,772]
	x = Batch_Size
	y = Labor_Cost
	n = len(y)

	x_mean = np.mean(x)
	y_mean = np.mean(y)

	# Finding the Slope
	SSxy = np.sum([(xi-x_mean) * (yi-y_mean) for xi, yi in zip(x, y)])
	SSxx = np.sum([pow((xi-x_mean), 2) for xi in x])
	b1 = SSxy/SSxx

	# FInding the y-intercept
	b0 = y_mean - (b1*x_mean)

	# Getting the point estimation based on b0 and b1 for the given Batch_Size 
	Labor_Cost_Point_Estimation = [b0 + b1*xi for xi in x]

	# Total variation of the given y from the mean(y)
	SSyy = np.sum([pow((yi-y_mean),2) for yi in y])

	# Getting the Sum of Squared Error using the error fiunction (e) e = y - (b0 + b1(x)) where (y is the true value, e is the min squared error and x is the Batch_Size data point)
	SSE = np.sum([pow((y_true-y_est), 2) for y_est, y_true in zip(Labor_Cost_Point_Estimation, y)])
	MSE =SSE/(n-2)  # Also called as "variance" or even "s_square"

	# Getting the Sum of Square Regression using the estimated point and their variance fromt he mean (y)
	SSR = np.sum([pow((y_est-y_mean),2) for y_est in Labor_Cost_Point_Estimation])

	# Getting the total error or Total Variance (The Sum of Squared error + The Sum of Squared Regression error)
	SST = SSE + SSR 

	# Correlation cooficient between X and Y
	corr_coef = np.corrcoef(x, y)

	batch_size = 60
	Labor_cost_Point_Estimation_for_given_barch_size = b0 + b1*batch_size

	print 'The given data point of the Batch_Size are: ', x
	print 'The respective data points of Labor_Cost are: ', y
	print 'The Mean value of the (Independent variable) Batch_Size is: ', x_mean
	print 'The Mean value of the (Dependent variable) Labor_Cost is: ', y_mean
	print 'The y-intercept and the slope are : ', b0, b1
	print 'The Point Estimation of Labor cost for 60 Batch Size is: ', Labor_cost_Point_Estimation_for_given_barch_size
	print ''
	print 'Sum of Total variation of the obsoved x and y from their respective mean (SSxy): ', SSxy
	print 'Sum of Total variation of the obsoved x its respective mean (SSxx): ', SSxx
	print 'Sum of Total variation of the obsoved y its respective mean (SSyy): ', SSyy
	print 'The Sum of Squared error (SSE) is: ', SSE
	print 'The Mean of Squared error (MSE) is: ', MSE, pow(MSE,0.5)
	print 'The Sum of Square Regression Error (SSR) is: ', SSR
	print 'The Sum of Square Total Error (SST): ', SST
	print 'The Estimated Labor_Cost is: ', Labor_Cost_Point_Estimation
	print 'The Cor-relation cofficient of the Batch_Size and Labor_Cost is: ', corr_coef
	# print 'The Point estimation for GPA 3.25 is: ', No_of_Minutes_Point_Estimation
	# print 'The Point Prediction for GPA 3.25 is: ', Salary_Point_Prediction_for_given_gpa


	# Ploting Graph
	plt.ion()
	plt.figure(1)
	plt.subplot(211)
	plt.plot(x,y,"ro")
	plt.axis([min(x)-2, max(x)+2, min(y)-2, max(y)+2])

	plt.subplot(212)
	plt.plot(x,y,"ro")
	plt.plot(x, Labor_Cost_Point_Estimation, "-")
	plt.axhline(y=y_mean, color = 'r')
	plt.axis([min(x)-2, max(x)+2, min(y)-2, max(y)+2])

	plt.show("wait")

Salary_data()