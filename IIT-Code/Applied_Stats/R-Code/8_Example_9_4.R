library(forecast)
dat<-read.table('~/Dropbox/Teaching/MATH 484:564/Data/Chapter 9/t9-4 viscosity.txt',header=T)
y<-ts(dat$y)
plot.ts(y)

#SAC: 
acf(y,lag.max=24,type='correlation',plot=TRUE)

#SPAC
pacf(y,lag.max=24,plot=TRUE)

#SAC dies down fairly quickly in a damped sine-wave fashion, so stationary. SPAC has spikes at lags 1 and 2 and cuts off after lag 2. It seems that AR(p) is the best model

ar_2<-arima(y,order=c(2,s,0))
y.forecasts <- forecast.Arima(ar_2, h=3)
plot.forecast(y.forecasts)