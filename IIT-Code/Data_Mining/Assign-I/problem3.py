
from __future__ import division
import numpy as np
import pandas as pd
import os, sys
from sklearn import decomposition



path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path2 = os.path.abspath(os.path.join(path1, ".."))
sys.path.append(path1)
sys.path.append(path2)
import config
from Tools import load_data_from_orange

# Tool Information Gain

conf_dict = config.get_config_settings()
df_data = load_data_from_orange(conf_dict['iris_data'])



# creating numpy attribute matrix to perform PCA and excluding the class label
attr = df_data.ix[:,0:df_data.shape[1]-1].values
# print attr
labels = df_data.ix[:,-1].values
labels_unq =  np.unique(np.array(labels))
# print labels_unq


##################################################################################################################################################


# Standarizing the data set
from sklearn.preprocessing import StandardScaler
attr_std = StandardScaler().fit_transform(attr)

# Performing PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(attr_std)
principal_component1 = principal_components[:,0]
principal_component2 = principal_components[:,1]

# Plot data
def plot1(X, Y, legend_label=None, xlabel=None, ylabel=None):
	import matplotlib
	# matplotlib.use('TkAgg')    # Since the matplotlib backend doesn't know we have to manually implement it 
	from matplotlib import pyplot as plt
	import seaborn as sns

	color_list = sns.color_palette("Set1", 10)
	plt.ion()
	with plt.style.context('seaborn-whitegrid'):
	    plt.figure(figsize=(6, 4))
	    color_list = color_list[0:len(legend_label)]
	    for label, color in zip(legend_label,color_list):
	        plt.scatter(X[labels==label],
	                    Y[labels==label],
	                    label=label,
	                    c=color)
	    if xlabel and ylabel:
		    plt.xlabel('Principal Component 1')
		    plt.ylabel('Principal Component 2')
	    plt.legend(loc='upper right')
	    plt.tight_layout()
	    plt.show('wait')

print(pca.explained_variance_ratio_)
plot1(principal_component1, principal_component2, legend_label=labels_unq, xlabel='Principal Component 1', ylabel='Principal Component 2')



##################################################################################################################################################


print (df_data)
def plot_distribution_for_attributes():
	import matplotlib
	matplotlib.use('TkAgg')    # Since the matplotlib backend doesn't know we have to manually implement it 
	from matplotlib import pyplot as plt

	sepal_length_arr = np.array(df_data['sepal length'], dtype="float")
	sepal_width_arr = np.array(df_data['sepal width'], dtype="float")
	petal_length_arr = np.array(df_data['petal length'], dtype="float")
	petal_width_arr = np.array(df_data['petal width'], dtype="float")
	# Using yaxis to be normal distribution
	sepal_length_arr_y = [ (1/(pow((2*3.14),0.5)*np.std(sepal_length_arr))) * np.exp((-1) * pow((x-np.mean(sepal_length_arr)),2)/(2*np.std(sepal_length_arr))) for x in sepal_length_arr]
	sepal_width_arr_y = [ (1/(pow((2*3.14),0.5)*np.std(sepal_width_arr))) * np.exp((-1) * pow((x-np.mean(sepal_width_arr)),2)/(2*np.std(sepal_width_arr))) for x in sepal_width_arr]
	petal_length_arr_y = [ (1/(pow((2*3.14),0.5)*np.std(petal_length_arr))) * np.exp((-1) * pow((x-np.mean(petal_length_arr)),2)/(2*np.std(petal_length_arr))) for x in petal_length_arr]
	petal_width_arr_y = [ (1/(pow((2*3.14),0.5)*np.std(petal_width_arr))) * np.exp((-1) * pow((x-np.mean(petal_width_arr)),2)/(2*np.std(petal_width_arr))) for x in petal_width_arr]
	plt.figure()
	plt.subplot(411)
	plt.plot(sepal_length_arr, sepal_length_arr_y, '.')
	plt.xlabel('sepal_length')
	plt.subplot(412)
	plt.plot(sepal_width_arr, sepal_width_arr_y, '.')
	plt.xlabel('sepal_width')
	plt.subplot(413)
	plt.plot(petal_length_arr, petal_length_arr_y, '.')
	plt.xlabel('petal_length')
	plt.subplot(414)
	plt.plot(petal_width_arr, petal_width_arr_y, '.')
	plt.xlabel('petal_width')
	plt.show('wait')
	


# plot_distribution_for_attributes()


##################################################################################################################################################

# def plot2(X, Y, legend_label=None, xlabel=None, ylabel=None):
# 	import matplotlib
# 	# matplotlib.use('TkAgg')    # Since the matplotlib backend doesn't know we have to manually implement it 
# 	from matplotlib import pyplot as plt
# 	import seaborn as sns

# 	color_list = sns.color_palette("Set1", 10)
# 	plt.ion()
# 	with plt.style.context('seaborn-whitegrid'):
# 	    plt.figure(figsize=(6, 4))
# 	    color_list = color_list[0:len(legend_label)]
# 	    for label, color in zip(legend_label,color_list):
# 	        plt.scatter(X[labels==label],
# 	                    Y[labels==label],
# 	                    label=label,
# 	                    c=color)
# 	    if xlabel and ylabel:
# 		    plt.xlabel('Principal Component 1')
# 		    plt.ylabel('Principal Component 2')
# 	    plt.legend(loc='upper right')
# 	    plt.tight_layout()
# 	    plt.show('wait')

# plot1(principal_component1, principal_component2, legend_label=['iris-versicolor', 'iris-virginica'], xlabel='Principal Component 1', ylabel='Principal Component 2')


# def plot2():
