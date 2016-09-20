dat<-read.table('~/Dropbox/Teaching/MATH 484:564/Data/Chapter 6/t6-4 hotel.txt',header=T)

y<-dat$y
n<-length(y)
Time<-1:n
plot(y,type='b',xlab='Time',ylab='Calculator sales')

y2<-log(y)
plot(y2,type='b',xlab='Time',ylab='Calculator sales')

H1<-sin(2*pi*Time/12)
H2<-cos(2*pi*Time/12)
H3<-sin(4*pi*Time/12)
H4<-cos(4*pi*Time/12)
trend<-lm(y2~Time+H1+H2+H3+H4)

y2.pred<-predict(trend, newdata=data.frame(Time=169, H1=sin(2*pi*169/12), H2=cos(2*pi*169/12), H3=sin(4*pi*169/12), H4=cos(4*pi*169/12)), interval='prediction')
y.pred<-data.frame(fit=exp(y2.pred[1]), lwr=exp(y2.pred[2]), upr=exp(y2.pred[3]))

d.stat<-sum((trend$residuals[1:(n-1)]-trend$residuals[2:n])^2)/sum(trend$residuals^2)

library(lmtest)
dwtest(trend, alternative = 'two.sided')




