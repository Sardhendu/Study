import csv

# datam = read.csv2("/Users/sam/All-Program/App-DataSet/Study/LoanStats3a.csv", header=TRUE, sep=",")
from pprint import pprint
i =0
with open('/Users/sam/All-Program/App-DataSet/Study/LoanStats3a_cleaned.csv', 'w') as wfh:
	with open('/Users/sam/All-Program/App-DataSet/Study/LoanStats3a.csv') as fh:
		while 1:
			line = fh.readline()
			# pprint(line)
			i+=1
			if i>1:
				wfh.write(line)

			# if i>1000:
			# 	break



	# reader = csv.reader(fh, skiprows)
	# for row in reader:
	# 	pprint (row)
	# 	break
# import pandas as pd

# dir = '/Users/sam/All-Program/App-DataSet/Study/LoanStats3a.csv'
# # df = pd.DataFrame.from_csv(dir, header=0, sep=',')

# df = pd.read_csv(dir, index_col=0)