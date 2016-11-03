# datam = read.csv2("/Users/sam/All-Program/App-DataSet/Study/LoanStats3a.csv", header=TRUE, sep=",")

datam = read.csv2("/Users/sam/All-Program/App-DataSet/Study/LoanStats3a_cleaned 1000.csv", header=TRUE, sep=",")

colnames(datam)

datam['addr_state']
datam['all_util']
datam['bc_util']
#loan_amnt
#int_rate
#grade


#python.Project_ASRA.call( "concat", a, b)
system('python -c /Users/sam/All-Program/App/Study/IIT-Code/Applied_Stats/Project/Project-ASRA.ipynb')





command = "python"
path2script='/Users/sam/All-Program/App/Study/IIT-Code/Applied_Stats/Project/Project-ASRA.py'
string = "3523462---12413415---4577678---7967956---5456439"
pattern = "---"
args = c(string, pattern)
allArgs = c(path2script, args)
output = system2(command, args=allArgs, stdout=TRUE)
print(paste("The Substrings are:\n", output))