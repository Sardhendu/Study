sales_territory<-read.table("../sam/All-Program/App-DataSet/R-Code/Chapter 5/ASCII comma/t5-1 sales territory complete.txt",header=T,sep=',')

# Lm is used to perform multiple Linear Regression
sales_territory

# Select a subset of the dataset
attributes <- c("Sales", "Time", "MktPoten", "Adver")
attributes
sales_territory2 <- sales_territory[attributes]
sales_territory2

# Get the summary Statistic of each attributes in sales_territory2
# Plotting the subset
plot(sales_territory2)

# Fit a Multiple Linear Regression for the first three variables
fit2<-lm(formula=Sales~ Time+MktPoten, data=sales_territory2)
summary(fit2)    # Get the summary of the fitted model

# Fit the Multiple Linear Regression for the first column
fit0<-lm(Sales~1,data=sales_territory)
summary(fit0)
fit<-lm(Sales~., data=sales_territory)
summary(fit)

scope<-list(upper=Sales~Time+MktPoten+Adver+MktShare+Change+Accts+WkLoad+Rating, lower=Sales~1)


# 
fit.forward<-step(fit0,direction='forward',scope=scope)
