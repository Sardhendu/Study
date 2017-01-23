# Adapted from http://dept.stat.lsa.umich.edu/~kshedden/Courses/Stat600/Notes/R_introduction.pdf
# Defining a variable in R
# Best practice is to use the <- assignment symbol
x <- 5
x = 5

# Creating a vector
# c stands for concatenate
myFirstVector <- c(2, 4, 5, "frog", "cat")

# Creating a vector of values from 1 to 10
tenItemVector <- c(1:10)
# R also supports creating more elaborate sequences
z = seq(3, 8, by=2)


# Creating a matrix
# By default, matrices are filled by column
x = matrix(c(2, 3, 4, 5, 6, 7), nrow=3, ncol=2)
x = matrix(c(2, 3, 4, 5, 6, 7), nrow=3, ncol=2, byrow=TRUE)

# Easy to get more information about anything in R
help(matrix)
?matrix
??matrices
print(x)


# R supports all the usual scalar arithmatic operations
x = 5
y = 7
z = x + y
a = y - x
w = a*z
x = 5^2
y = 23 %% 5

# R allows variables to be modified in place
x = 5
x = x + 1

# R supports all of the typical rounding functions
v = ceiling(3.8)
w = floor(3.8)
x = ceiling(-3.8)
y = trunc(-3.8)
z = round(-3.8)


# Additional mathematical functions are easily available
w = sqrt(2)
x = exp(3)
y = log(x)
z = tan(pi/3)


# Boolean expressions
3 + 2 < 5
10 - 4 > 5
# Note the double equal sign for the evaluation of equality
10 + 4 == 7 + 7

# The & (and) operator is TRUE only if the expressions on both sides of the
# operator are TRUE
(3 < 5) & (2 > 0)
(2 < 3) & (5 > 5)

# The | (or) operator is TRUE if at least one of the expressions surrounding it is
# TRUE.
(3 < 5) | (2 > 3)
(2 < 1) | (5 > 5)

# R provides a very useful NOT operator, !
5 != (3 + 2)
5 == (3 + 2)

# Boolean operations on vectors are element-wise
myVector <- 1:10
myVector <= 5
sum(myVector <= 5)


# R supports looping
# Note that R loops CAN be extremely slow. Best practice is to use the set of
# apply functions whenever possible
x = 0
for (i in 1:10)
{
  x = x + i
}


# As expected, R supports while loops
n = 1 ## Don’t forget
x = 0 ## to initialize
while (x < 5)
{
  x = x + 1/n ## Note that the order of these two statements
  n = n+1 ## in the block is important.
}

# R supports conditional expressions
y = 7
if (y < 10) { x = 2 }
else { x = 1 }

A = 0
B = 0
C = 0
D = 0
for (k in (1:100))
{
  ## An if construct.
  if ((k %% 2 == 1) & (k < 50)) { A = A + k }
  else if ((k %% 2 == 1) & (k >= 50)) { B = B + k }
  else { C = C + k }

  # Note that if's need not always have an else
  ## An independent if construct.
  if (k >= 50) { D = D + k }
}


# Functions are easily defined in R
myFirstFunc <- function(x, y){
  # Best practice is to explicitly specify "return", though this is not required
  return(x * y)
}
myFirstFunc(2, 3)

myFirstFunc <- function(x, y){
  x * y
  # Note how the absence of the return statement still allows the function to
  # run, but for large and complicated functions it adds confusion.
  # tldr: add "return" to every function to create!
  }


# Default values can be specified
newFunc <- function(x = 5, y = 10){
  return(x * y)
}
newFunc()

# Something somewhat unique about R is that function defaults can be executable expressions
# Use this sparingly as this can lead to difficult to read code or unexpected
# behvaior
b <- 5
newNewfunc <- function(x = 3 * b){
  return(x)
}
newNewfunc()


# List material adapted from http://faculty.nps.edu/sebuttre/home/R/lists.html
# R uses lists as the data type for many useful things
mylist <- list (a = 1:5, b = "Hi There", c = function(x) x * sin(x))

mylist # we can see the contents of the list

# Obtain the first item from the list
mylist[1]

# Why does this fail?
mylist[1] * 5
# If we want to get the items out in their original form, we use [[]]
mylist[[1]]
mylist[[1]] * 5
# Can also refer by name
mylist$a
# Or like this
mylist['a']

# What's the second element of the item named "a"?
mylist$a[2]

# We can use negative indices too!
# Give me everything from "a" except the second element
mylist$a[-2]


# Let's add a new item to our list, in a slot we'll call "d"
mylist$d <- "New item"
mylist

# To delete an item, we simply assign it to NULL
mylist$b <- NULL
mylist

# Data frames are extremely useful data structures for analysis
# Adopted from http://www.r-tutor.com/r-introduction/data-frame
n = c(2, 3, 5)
s = c("aa", "bb", "cc")
b = c(TRUE, FALSE, TRUE)
df = data.frame(n, s, b)

# You can easily determine the number of rows and columns in a data.frame
nrow(df)
ncol(df)

# R provides some built in data.frames for practice
df <- mtcars

# R supports unix-like commands for head and tail to inspect the contents of a
# data.frame
head(df)
tail(df)


# data.frames support named columns
names(df)

# You can select from data.frames by row, column or both
df[1:3,] # Gets the first 3 rows and all columns of the data.frame
df[1:3, 'mpg']


# You can select multiple columns using a vector of column names
df[1:3, c('mpg', 'cyl')]

# You can perform element-wise mathematical operations on data.frames
df[,'mpg'] * 5

# To create a new column in a data.frame, simple assign a vector to the new
# named column
df[, 'mpgSquared'] <- df[,'mpg'] ** 2

# You can select data.frame elements via boolean vectors
# This returns only the rows where mpg is less than or equal 20
df[df[,'mpg'] <= 20,]

# Best practice is to use the apply functions to perform operations on
# every element
?apply
?lappy
?sapply
sapply()

sapply(X = df[,'mpg'], FUN = sum)



# R functionality is greatly extended by installing packages
install.packages('gdata')

# Installed packages are loaded via the library() command
library('gdata')
