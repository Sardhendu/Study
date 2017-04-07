# CSP 571 Homework 4


# 1. Please write a function called backwards() that implements the
# backward selection algorithm using AIC.



# 2. Download the credit card default data set from the UCI machine learning
# repository. https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
# Load the data into R.

dataDir <- "/Users/sam/All-Program/App-DataSet/Study/Data-Prep-Analysis/default_of_credit_card_clients.xls"

credit.data <- read.xls (dataDir, sheet = 1,skip=1, header = TRUE)
head(credit.data)

# Remove the ID column:
credit.data <- subset(credit.data, select=-c(ID))
head(credit.data)

# 3. Identify all the relevant categorical, numeric, and logical variables.

credit.numericCols <- names(which(sapply(credit.data, is.numeric)))
credit.nominalCols <- names(which(sapply(credit.data, is.factor)))
print (nrow(credit.data))
print (ncol(credit.data))
print (credit.numericCols)
print (credit.nominalCols)

# Check if data is missing
which(is.na(credit.data))


# 4. Perform all required EDA on this data set.
plot(credit.data)

# 5.Build a logistic regression model to determine whether or not a
# customer defaulted. Use all of the variables. Validate the model on a
# test data set. Use the comments to discuss the performance of the model.


# 6. Using forward selection, determine the best model.


# 7. Using the backwards selection function you implemented in #1
# , determine the best model.


# 8. Run an implementation of backwards selection found in an R package on this
# data set. Discuss any differences between the results of this implementation
# and your implemnetation in question 7.


# 9. Run lasso regression on the data set. Briefly discuss how you determined
# the appropriate tuning parameters.


# 10. Run ridge regression on the data set. Briefly discuss how you determined
# the appropriate tuning parameters.

# 11. Run naive bayes on the data set.


# 12. Build a decision tree to classify the customers as defaulted
# or not-defaulted. Plot the resulting tree. Discuss whether you
# feel this is a good model.


# 13. Build a random forest model and apply it to classify the test data set.


# 14. Discuss the comparative performance of all of the models used. How should
# we determine which is best? Provide justification for your answer.