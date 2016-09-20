# Explanation of this example and more on Logistic regression: 
# Go to: http://ww2.coastal.edu/kingw/statistics/R-tutorials/logistic.html


file = "http://ww2.coastal.edu/kingw/statistics/R-tutorials/text/gorilla.csv"
read.csv(file) -> gorilla
str(gorilla)

cor(gorilla)
with(gorilla, tapply(W, seen, mean))
with(gorilla, tapply(C, seen, mean))
with(gorilla, tapply(CW, seen, mean))

glm.out = glm(seen ~ W*C*CW, family=binomial(logit), data=gorilla)
summary(glm.out)

scope=list(lower=seen~1, upper=seen~W*C*CW)
fit.null<-glm(seen ~ 1, family=binomial(logit), data=gorilla)
fit.step<-step(fit.null, direction='forward',data=gorilla, scope=scope,k=log(nrow(gorilla)))

anova(glm.out, test="Chisq")
1 - pchisq(8.157, df=7)
plot(glm.out$fitted)
abline(v=30.5,col="red")
abline(h=.3,col="green")
abline(h=.5,col="green")
text(15,.9,"seen = 0")
text(40,.9,"seen = 1")