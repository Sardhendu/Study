# Store location example

dat<-read.table('../Data/Chapter\ 4/t4-9\ store\ location\ complete.txt',header=T,sep=',')

dat$loca<-c(rep('street',5),rep('mall',5),rep('downtown',5))
dat$loca<-as.factor(dat$loca)

summary(dat)
# two ways

fit1<-lm(y~x+DM+DD, data=dat)
summary(fit1)

fit2<-lm(y~x,data=dat)
summary(fit2)
anova(fit1,fit2)

fit3<-lm(y~x+DM,data=dat)
summary(fit3)
anova(fit1,fit3)

fit4<-lm(y~x+DD,data=dat)
summary(fit4)
anova(fit1,fit4)
