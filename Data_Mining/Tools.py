"""
	Module Creator	:	Sardhendu Mishra
	Aim				:	Small tools for Data Analysis
"""

from __future__ import division
import numpy as np
import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


# Tool Information Gain

conf_dict = config.get_config_settings()
# print conf_dict
# a = pd.read_table(conf_dict['titanic_data'], sep='\t', lineterminator='\r')

def load_data_from_orange(dir_path):
	data_table = pd.DataFrame.from_csv(dir_path, sep='\t', header=0, index_col=None)
	data_table = data_table.ix[2:]
	return data_table



class Entropy_InfoGain():

	def cal_entropy(self, pi_values):
		print ('The pi values are: ', pi_values)
		entropy = np.sum([(-pi * np.log2(pi)) for pi in pi_values if pi!=0])
		return entropy

	def cal_information_gain(self, child, parent):

		df1 = pd.DataFrame({'attr': child, 'class': parent})
		
		unq_attr = np.unique(df1['attr'])
		unq_class = np.unique(df1['class'])
		tot_data = len(df1)

		parent_entropy = self.cal_entropy([len(np.where(parent==class_name)[0])/tot_data for class_name in unq_class])

		no_of_datapoints = len(child)

		df_att_entropy = pd.DataFrame(columns=['att_value', 'weight_of_att_value', 'entropy'])
		for i, att_val in enumerate(unq_attr):
			df2 = df1.loc[df1['attr'] == att_val]			# get the dataframe were t
			p_class_given_att_vals = [len(df2['class'].loc[df2['class'] == class_val])/len(df2) for class_val in unq_class]
			entropy_for_attr_val = self.cal_entropy(p_class_given_att_vals)
			df_att_entropy.loc[i] = [att_val, len(df2)/no_of_datapoints, entropy_for_attr_val]

		# Information gain = Entropy(parent) - weighted_average(entropy(children))
		weighted_avg_entropy_of_child = sum([df_att_entropy.loc[i]['weight_of_att_value'] * df_att_entropy.loc[i]['entropy'] for i in range(0,len(df_att_entropy))])

		# Information_Gain = parent_entropy - weighted_avg_entropy_of_child
		information_gain = parent_entropy - weighted_avg_entropy_of_child

		print ('The unique classes mentioned are: ', unq_class)
		print ('The unique attribute values are: ', unq_attr)
		print ('The entropy of the Parent is: ', parent_entropy)
		print ('The entropy of attributes values based on the class is: \n', df_att_entropy)
		print ('The total weighted entropy of the feature attribute is: ', weighted_avg_entropy_of_child)
		print ('The Information gain between the feature atribute and the class is: ', information_gain)





# def gini_index():



# data = load_data_from_orange(conf_dict['titanic_data'])
# Entropy_InfoGain().cal_information_gain(data['sex'], data['survived'])

