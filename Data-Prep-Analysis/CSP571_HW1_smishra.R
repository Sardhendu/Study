# CSP/MATH 571
# Homework 1




# Question 1: Create a variable named "myName" and assign to have a value of your
# preferred name. Create a varaible named "myEmail" and assign it to have a value
# of your email.
myName <- 'Sardhendu'
myName
myEmail <- 'smishra13@hawk.iit.edu'
myEmail



# Question 2: Create a vector of integers from 100 to 10000 (inclusive). Assign
# the variable myVector. Return the sum, min, max, and median of this vector and assign it
# below.
myVector <- c(100:10000)
q2Sum <- sum(myVector)
q2Min <- min(myVector)
q2Max <- max(myVector)
q2Median <- median(myVector)




# Question 3: Write a function that accepts a number as an input returns
# TRUE if that number is divisible by 127 FALSE if that number is not divisible
# by 127.  For example, divis(127*5) should return TRUE and divis(80)
# should return FALSE. Hint: %% is the modulo operator in R.
divis = function(inp_num, div_by=127){
  if (inp_num%%div_by == 0){
    return (TRUE)
  }
  else{
    return (FALSE)
  }
}

divis(127*5)
divis(80)





# Question 4: Using the function you wrote for Question 3 and the vector you
# defined in Question 2, deterine how many integers between 100 and 10000 are
# divisible by 127. Assign it to the variable below.
countDivis <- sum(sapply(myVector, FUN=divis))
countDivis
# sapply takes a list as an input and performs the operation as suggested in the FUN and return return s the list.

#bb <- c(127,128,129,127*2,127*3)
#a = sapply(bb, FUN=divis)





# Question 5: Using the vector of names below, write code to return the 9
# last name in the vector.
names <- c("Kermit Chacko",
           "Eleonore Chien",
           "Genny Layne",
           "Willene Chausse",
           "Taylor Lyttle",
           "Tillie Vowell",
           "Carlyn Tisdale",
           "Antione Roddy",
           "Zula Lapp",
           "Delphia Strandberg",
           "Barry Brake",
           "Warren Hitchings",
           "Krista Alto",
           "Stephani Kempf",
           "Sebastian Esper",
           "Mariela Hibner",
           "Torrie Kyler")

# Returning the 9th last name
ninthLastName <- tail(strsplit(names[9], " ")[[1]], n=1)
ninthLastName
# Returning the last 9 names
LastnineNames <- names[-1:-(length(names)-9)]
LastnineNames

# names[-1]





# Question 6: Using the vector "names" from Question 5, write code to determine how many last names start with L.
countLastNameStartsWithL <- sum(grepl(" L", names))  
countLastNameStartsWithL
# Using regex operation, grepl will capture any string that that has a space before, provided the string must start with a "L". The case would however fail when a name contains 3 words. A much better approach is given below
countLastNameStartsWithL <- sum(sapply(strsplit(names," "), function(a) grepl("L", tail(a, n=1))))
countLastNameStartsWithL
# The difference is that we use sapply to loop through each element and then apply the function to find the last word in a name, then we do a simple Regex Match and finally Sum. 





# Question 7: Using the vector "names" from Question 5, write code to create a
# list that allows the user to input a first name and retrieve the last name.
# For example, nameMap["Krista"] should return "Alto".
dict_namemap <- sapply(strsplit(names, " "), function(a) c(a[1], tail(a, n=1))) # extracts all the last name
nameMap <- dict_namemap[2,]   # Retrieves the list with the last name
names(nameMap) <- dict_namemap[1,] # Retrieves the list with the first name and maps it to the last name.
nameMap['Krista']
# The code just runs sapply once and forms two stacked list of the first name and last name and then maps each first name to the last name.

mapping <- c(
  'A40'='car (new)',
  'A41'='car (used)',
  'A42'='furniture/equipment',
  'A43'='radio/television',
  'A44'='domestic appliances'
)




# Question 8: Load in the "Adult" data set from the UCI Machine Learning
# Repository. http://archive.ics.uci.edu/ml/datasets/Adult
# Load this into a dataframe. Rename the variables to be the proper names
# listed on the website. Name the income attribute (">50K", "<=50K") to be
# incomeLevel
data <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep=',')
# Renaming the variables
data <- data.frame(data=data)
#colnames(data) = c('age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week', 'native-country','incomeLevel')








# Question 9: Create a new variable called workSector. Label all government
# employees as "government", all self-employeed employees as "selfEmployed",
# all Private employees as "Private" and everyone else as "Other".

library(gdata)
unq_workclass = unique(data['workclass'])
gov <- c('State-gov', 'Federal-gov', 'Local-gov')
self <- c('Self-emp-not-inc', 'Self-emp-inc')
prv <- c('Private')

# The below code will find all the index position where the value belongs to either government, self-employed or Private 
gov_indx <- which(!is.na(match(trim(data$workclass), gov)))
self_indx <- which(!is.na(match(trim(data$workclass), self)))
prv_indx <- which(!is.na(match(trim(data$workclass), prv)))

# Creating a new column and inserting the respective values
data[gov_indx, 'workSector'] <- 'government'
data[self_indx, 'workSector'] <- 'selfEmployed'
data[prv_indx, 'workSector'] <- 'Private'
data[which(is.na(data$workSector)), 'workSector'] <- 'Other'
head(data)





# Question 10: Create a histogram of the 'age'. Hint: You'll need to convert
# age to be numeric first. Save this histogram and include it with your
# submission
# Storing the age column into a list with dtype as numeric
age <- as.numeric(as.character(data$age))
hist(age) # Plotting the histogram





# Question 11: Determine the top 3 occupations with the highest average hours-per-week
# Hint: One way to do this is to use tapply
library(sqldf)
top3_Occup <- sqldf("select occupation, avg(hours_per_week) as h_avg from data group by occupation order by h_avg desc limit 3")['occupation']
top3_Occup
# Using SQL like query to find the top 3 occupation that

