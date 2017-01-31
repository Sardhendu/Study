"""
	Module Creator	:	Sardhendu Mishra
	Aim				:	Small tools for Data Analysis
"""

from __future__ import division
import numpy as np
import pandas as pd
import os, sys

path1 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path2 = os.path.abspath(os.path.join(path1, ".."))
sys.path.append(path1)
sys.path.append(path2)
import config
from Tools import load_data_from_orange

# Tool Information Gain

conf_dict = config.get_config_settings()
data = load_data_from_orange(conf_dict['auto_mpg_data'])
# print(data.DESCR)


def provide_column_name(column_name):
	new_array_with_nan = np.array(data[column_name])
	indices = np.where(np.array(data[column_name]) == "?")[0]
	print (indices)
	new_array_with_nan[indices] = np.NaN
	return new_array_with_nan


# print data
new_array_with_nan = provide_column_name('horsepower')	
data['horsepower'] = new_array_with_nan

# print (data.DESCR)
print (data.describe())
# print data.describe()

# print len(new_array_with_nan)

# ###################
# # Statistics with Nan
# ###################
# new_array_withon_Nan =  np.array([i for i in new_array_with_nan if str(i)!='nan'], dtype='float')
# print len(new_array_withon_Nan)
# print new_array_withon_Nan
# # The data is skewed towards the right
# mean1 = np.mean(new_array_withon_Nan)
# variance1 = np.var(new_array_withon_Nan)
# print mean1 , variance1

# print''
# ###################
# # Statistics without Nan
# ###################
# new_array_mean_replaces_Nan =  np.array([i if str(i)!='nan' else mean1 for i in new_array_with_nan], dtype='float')
# print len(new_array_mean_replaces_Nan)
# print new_array_mean_replaces_Nan

# mean2 = np.ceil(np.mean(new_array_mean_replaces_Nan))
# variance2 = np.var(new_array_mean_replaces_Nan)
# print mean2 , variance2


# ###################
# # Pot data
# ###################
# import matplotlib
# matplotlib.use('TkAgg')    # Since the matplotlib backend doesn't know we have to manually implement it 
# from matplotlib import pyplot as plt

# plt.ion()
# plt.figure(1)
# plt.subplot(211)
# plt.plot(new_array_withon_Nan, np.zeros(len(new_array_withon_Nan)), '.')

# plt.subplot(212)
# plt.plot(new_array_mean_replaces_Nan, np.zeros(len(new_array_mean_replaces_Nan)), '.')
# plt.show('wait')