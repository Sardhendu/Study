
# Load the data
dat = read.table("../sam/All-Program/App-DataSet/R-Code/Chapter 4/ASCII comma/t4-9 store location complete.txt", header=T, sep=',')

# Add the Categorical term
dat$loca<-c(rep('street',5),rep('mall',5),rep('downtown',5))

# Inorder to make it representable we add a location feature.
dat$loca<-as.factor(dat$loca)


# Now let us see how different models do using the partial F Test:

# For complete model using Dummy Variables
fit <- lm(formula=y~x+DM+DD, data=dat)
summary(fit)
# The above model is not very good as the t-statistics for the dummy variable shows a value of 0.178 making the vaiable insignificant.

# Fitting a different model without th DD term
fit2 <- lm(formula=y~x+DM, data=dat)
summary(fit2)
# In this case all the independent variable seems significant, however the intercept term has a t-statistic of 0.0597. This is okay.

# Fitting the model wothout the dumy variables
fit3 <- lm(formula=y~x, data=dat)
summary(fit3)



# NOW Testing the statistics and comparing models using ANOVA outcomes.
anova(fit,fit2)
anova(fit,fit3)
anova(fit2,fit3)
