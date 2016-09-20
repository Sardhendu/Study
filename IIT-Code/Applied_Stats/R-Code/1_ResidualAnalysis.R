library(HH)
library(leaps)
sales_territory<-read.table("~/Dropbox/Teaching/MATH 484:564/Data/Chapter 5/t5-1 sales territory complete.txt",header=T,sep=',')

fit0<-lm(Sales~1,data=sales_territory)
fit<-lm(Sales~., data=sales_territory)
scope<-list(upper=Sales~Time+MktPoten+Adver+MktShare+Change+Accts+WkLoad+Rating, lower=Sales~1)
fit.forward<-step(fit0,direction='forward',scope=scope)
fit.backward<-step(fit,direction='backward',scope=scope)
fit.both<-step(fit,direction='both',scope=scope)

fit<-fit.backward

#Residual plot
par(mfrow=c(2,3))
plot(y=fit$residuals,x=fit$fitted.values,xlab=expression(hat(y)),ylab='residuals')
plot(y=fit$residuals,x=sales_territory$Time,xlab=expression(Time),ylab='residuals')
plot(y=fit$residuals,x=sales_territory$MktPoten,xlab=expression(MktPoten),ylab='residuals')
plot(y=fit$residuals,x=sales_territory$Adver,xlab=expression(Adver),ylab='residuals')
plot(y=fit$residuals,x=sales_territory$MktShare,xlab=expression(MktShare),ylab='residuals')
plot(y=fit$residuals,x=sales_territory$Change,xlab=expression(Change),ylab='residuals')

#QQ plot
qqnorm(fit$residuals)
qqline(fit$residuals)
