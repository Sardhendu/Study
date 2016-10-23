#det_dat <- read.table("../sam/All-Program/App-DataSet/R-Code/Chapter 4/ASCII comma/t4-6 fresh detergent 2.txt", header=T, sep=",")

det_dat_2<- read.table("../sam/All-Program/App-DataSet/R-Code/Chapter 4/ASCII comma/t4-5 gas additive.txt", header = T, sep=",")

# Get the summary as usual, helps you understand the limits and range of the dataset.
summary(det_dat_2)

# plot all the data to see how correlated the data points are:
plot(det_dat_2)
# The plot looks like a quadratic, but here the data is 2D so it is easy to vizualize, however for a higher dimensional dataset, we plot the residues and see if there is a quadratic patters.

# First we fit the model
names(det_dat_2)
fit<-lm(Mileage~Units, data=det_dat_2)
plot(fit)
summary(fit)
# After we observe this plot we can say that the redidue is a quadratic function, which means there is some quadratic variance not captured by our model. SO we add a Quadratic term to out model.

# Fitting the model with quadratic term:
fit_q<-lm(Mileage~Units+I(Units^2), data=det_dat_2)
plot(fit_q)
summary(fit_q)
# Now if we see the residue plot, we see that the erros are most likely equally spaces arrouund the mea, which means that we have a good model. Moreever, the p-values are 0 which further bolsters the fact tht the model is good

# Now we make some Prediction using the fitted model:
testx<-seq(from=0,to=5,length=300)   # Equivallent of linspace in python
test.det_dat <- data.frame(Units=testx)
length(testx)
pred<-predict(fit_q,newdata=test.det_dat)
length(pred)

#plot(dat$x,dat$y,type='point',xlab='x',ylab='y')
plot(det_dat_2$Units,det_dat_2$Mileage, type='point', xlab='Units', ylab="Mileage")
lines(test.det_dat$Units, pred, lty='dashed', col='red')


