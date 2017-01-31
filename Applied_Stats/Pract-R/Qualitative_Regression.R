# Store location example
dat<-read.table("../sam/All-Program/App-DataSet/Study/IIT-Code/Applied_Stats/Chapter 4/ASCII comma/t4-9 store location complete.txt",header=T,sep=',')

# The data loaded has two Dummy variable DM and DD, where DM=1 when the transaction belongs
# to the store at Mall, DD=1 when the transaction belongs to the downtown store
# when both DM and DD are 0 then the transaction belongs to the store in the street
# Inorder to make it representable we add a location feature.
dat$loca<-c(rep('street',5),rep('mall',5),rep('downtown',5))

# We get the sumarry statistics:
summary(dat)

# We plot the Data exclusing the last column and the first column:
plot(dat[1:dim(dat)[2]-1])

# We encode the new column loca as a factor because by doing so we make sure that the column
# can be used in statistical modeling where they will be implemented correctly, that is they 
# will be assigned correct degrees of freedom and etc.
dat$loca<-as.factor(dat$loca)


# Model:
# y = B0 + B1x + B2Dm + B3Dd,   
# where Dm=1 when is in mall else Dm=0 and Dd=1 when the store is in downtown else Dd=0 
  
# two ways
fit1<-lm(y~x+DM+DD, data=dat)
summary(fit1)

# This piece of code converts the categorical column implicitely into collection of 
# dummy variable
fit2<-lm(y~x+loca,data=dat) 
summary(fit2)

# This code is equivallent of linspace in python
testx<-seq(from=99,to=248,length=50) 
# We just assign the test data to each of the qualitative attribute (street, mall, downtown)
# so that the point prediction uses the specific model for the given attribute
test.dat<-data.frame(x=rep(testx,3),loca=as.factor(c(rep('street',50),rep('mall',50),rep('downtown',50))))
pred<-predict(fit2,newdata=test.dat)

# below shows the plot
# If you see the plot, you see that the point prediction model as below:
#   For Street:   y_hat = B0 + B1x
#   For Mall:     y_hat = B0 + B1x +B2   
#   For Downtown: y_hat = B0 + B1x +B3
# are all parallel to each other, this says that their is no interaction between the variables
plot(dat$x,dat$y,type='point',xlab='x',ylab='y')
lines(testx,pred[test.dat$loca=='street'],lty='dashed')
lines(testx,pred[test.dat$loca=='mall'],lty='dashed',col='red')
lines(testx,pred[test.dat$loca=='downtown'],lty='dashed',col='blue')
legend('topleft',legend=c('street','mall','downtown'),lty=rep('dashed',3),col=c('black','red','blue'))




