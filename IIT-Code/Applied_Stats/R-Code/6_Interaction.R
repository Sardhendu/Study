# Bonner Frozen Foods

dat<-read.table('../Data/Chapter\ 4/t4-7\ bonner\ adtype.txt',header=T,sep=',')

fit1<-lm(Volume~RadioTV+Print,data=dat)
summary(fit1)
interaction.plot(as.factor(dat$RadioTV),as.factor(dat$Print),dat$Volume,type='b')

fit2<-lm(Volume~RadioTV*Print,data=dat)
summary(fit2)

fit2<-lm(Volume~RadioTV+Print+I(RadioTV*Print),data=dat)
