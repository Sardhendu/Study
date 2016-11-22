library(HH)
library(leaps)

sales_territory<-read.table("../sam/All-Program/App-DataSet/Study/IIT-Code/Applied_Stats/Chapter 5/ASCII comma/t5-1 sales territory complete.txt",header=T,sep=',')

pairs(~Sales+Time+MktPoten+Adver+MktShare+Change+Accts+WkLoad+Rating,data=sales_territory)
cor(sales_territory[,-1])
fit<-lm(Sales~., data=sales_territory)
vif(fit)

#Best subset Model Approach
best.fit<-regsubsets(x=sales_territory[,-1], y=sales_territory$Sales,data=sales_territory,nbest=10)
best.fit.results<-summary(best.fit)
plot(rowSums(best.fit.results$which), best.fit.results$cp, xlab='model size p+1', ylab='Mellow Cp')
model.size<-unique(rowSums(best.fit.results$which))
min.cp<-numeric(8)
for (i in 1:8) {
  min.cp[i]<-min(best.fit.results$cp[(rowSums(best.fit.results$which))==(i+1)])
}
lines(model.size,min.cp)

#Stepwise
fit0<-lm(Sales~1,data=sales_territory)
scope<-list(upper=Sales~Time+MktPoten+Adver+MktShare+Change+Accts+WkLoad+Rating, lower=Sales~1)
fit.forward<-step(fit0,direction='forward',scope=scope)
fit.backward<-step(fit,direction='backward',scope=scope)
fit.both<-step(fit,direction='both',scope=scope)


# http://stats.stackexchange.com/questions/162637/best-subset-selection-with-categorical-data

##### ROugh

#Loading Data:
sales_territory<-read.table("../sam/All-Program/App-DataSet/Study/IIT-Code/Applied_Stats/Chapter 5/ASCII comma/t5-1 sales territory complete.txt",header=T,sep=',')  

# Fitting the smallest model
fit0<-lm(Sales~1,data=sales_territory)

# Fit the largest model
fit<-lm(Sales~.,data=sales_territory)

# Building the scope
scope<-list(upper=Sales~Time+MktPoten+Adver+MktShare+Change+Accts+WkLoad+Rating, lower=Sales~1)

# Using step-wise Backward Plot (Default for AIC)
fit.backward<-step(fit,direction='backward',scope=scope)

# Now lets plot the resifue plot:
par(mfrow=c(2,3))
plot(fit$residuals, x=fit$fitted.values, xlab = expression(hat(y)), ylab='residuals')
plot(fit$residuals, x=sales_territory$Time, xlab = expression(Time), ylab='residuals')
plot(fit$residuals, x=sales_territory$MktPoten, xlab = expression(MktPoten), ylab='residuals')
plot(fit$residuals, x=sales_territory$Adver, xlab = expression(Adver), ylab='residuals')
plot(fit$residuals, x=sales_territory$MktShare, xlab = expression(MktShare), ylab='residuals')
plot(fit$residuals, x=sales_territory$Change, xlab = expression(Change), ylab='residuals')
plot(fit$residuals, x=sales_territory$Accts, xlab = expression(Accts), ylab='residuals')
plot(fit$residuals, x=sales_territory$WkLoad, xlab = expression(WkLoad), ylab='residuals')
plot(fit$residuals, x=sales_territory$Rating, xlab = expression(Rating), ylab='residuals')

# Do the QQ plot
qqnorm(fit$residuals)
qqline(fit$residuals)
