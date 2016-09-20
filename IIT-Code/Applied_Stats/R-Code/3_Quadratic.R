# The Gasoline Mileage Data

dat<-read.table('../Data/Chapter\ 4/t4-5\ gas\ additive.txt',header=T,sep=',')

fit1<-lm(Mileage~Units,data=dat)
summary(fit1)
plot(dat$Units, fit1$resid,type='p')

fit2<-lm(Mileage~Units+I(Units^2),data=dat)
summary(fit2)
plot(dat$Units, fit2$resid,type='p')

# Fresh Detergant 

detergant<-read.table('../Data/Chapter\ 4/t4-6.txt',header=T,sep=',')
pairs(~Demand+Price+IndPrice+AdvExp, data=detergant, main='Simple Scatterplot Matrix')

fit1<-lm(Demand~Price+IndPrice+AdvExp, data=detergant)
summary(fit1)

fit2<-lm(Demand~I(Price-IndPrice)+AdvExp, data=detergant)
summary(fit2)
plot(detergant$Price-detergant$IndPrice, fit2$resid, type='p')
plot(detergant$AdvExp, fit2$resid, type='p')

fit3<-lm(Demand~I(Price-IndPrice)+AdvExp+I(AdvExp^2), data=detergant)
summary(fit3)
plot(detergant$Price-detergant$IndPrice, fit3$resid, type='p')
plot(detergant$AdvExp, fit3$resid, type='p')

