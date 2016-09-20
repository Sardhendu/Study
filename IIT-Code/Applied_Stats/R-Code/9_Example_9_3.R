dat<-read.table('~/Dropbox/Teaching/MATH 484:564/Data/Chapter 9/t9-1 towel.txt',header=T)
y<-ts(dat$y)
plot.ts(y)

#SAC: 
acf(y,lag.max=24,type='correlation',plot=TRUE)

#SPAC
pacf(y,lag.max=24,plot=TRUE)

#SAC dies down very slow, not stationary. 

z<-diff(y,difference=1)
plot.ts(z)
#SAC: 
acf(z,lag.max=24,type='correlation',plot=TRUE)

#SPAC
pacf(z,lag.max=24,plot=TRUE)

ar_1<-arima(z,order=c(1,0,0))
arima_1_1<-arima(y,order=c(1,1,0)) # This model has smaller aic. 



