# Store location example

dat<-read.table('../Data/Chapter\ 4/t4-9\ store\ location\ complete.txt',header=T,sep=',')

dat$loca<-c(rep('street',5),rep('mall',5),rep('downtown',5))
dat$loca<-as.factor(dat$loca)

summary(dat)
# two ways

fit1<-lm(y~x+DM+DD, data=dat)
summary(fit1)

fit2<-lm(y~x+loca,data=dat)
summary(fit2)

testx<-seq(from=99,to=248,length=50)
test.dat<-data.frame(x=rep(testx,3),loca=as.factor(c(rep('street',50),rep('mall',50),rep('downtown',50))))
pred<-predict(fit2,newdata=test.dat)

plot(dat$x,dat$y,type='point',xlab='x',ylab='y')
lines(testx,pred[test.dat$loca=='street'],lty='dashed')
lines(testx,pred[test.dat$loca=='mall'],lty='dashed',col='red')
lines(testx,pred[test.dat$loca=='downtown'],lty='dashed',col='blue')
legend('topleft',legend=c('street','mall','downtown'),lty=rep('dashed',3),col=c('black','red','blue'))