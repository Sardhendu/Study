dat = read.table("../sam/All-Program/App-DataSet/R-Code/Chapter 4/ASCII comma/t4-9 store location complete.txt", header=T, sep=',')

# Get the Summary
summary(dat)

# Before appending the interaction term lets check if the interaction is okar to use:
# What we do is we plot the y and x for different location of stores (strees, mall and downtown)

# Covert the model into including the interaction term:
# The model will be y = B0 +B1x + B2Dm + B3Dd + B4xDm + B5xDd
# We have to build interaction columns:
dat$xDM<-dat["x"]*dat["DM"]
dat$xDD<-dat["x"]*dat["DD"]

# Adding the last category column:
dat$loca<-c(rep('street',5), rep('mall',5), rep('downtown',5))

# We encode the new column loca as a factor because by doing so we make sure that the column
# can be used in statistical modeling where they will be implemented correctly, that is they 
# will be assigned correct degrees of freedom and etc.
dat$loca<-as.factor(dat$loca)
#dat$xDM<-as.factor(dat$xDM)
#dat$xDD<-as.factor(dat$xDD)



# Recheck the sumarry 
sumarry(dat)

# Now we fit the model
fit1<-lm(y~x+DM+DD+xDM+xDD, data=dat)
