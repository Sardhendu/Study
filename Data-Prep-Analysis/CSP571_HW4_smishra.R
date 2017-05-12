# CSP 571 Homework 4
rm(list = ls())

require(gdata)
library(caret)
library(ggplot2)
library(corrplot)
library(grid)
library(gridExtra)
library(reshape)
library(ROCR)
library(glmnet)      # For Logistic/Lasso/Ridge
library(MASS)        # For Ridge Regression
library(e1071)       # For Naive Bayes
library(rpart)       # For Decision Tree's
library(rpart.plot)
library(RColorBrewer)
library(randomForest)

# fancyRpartPlot(fit)
# fancyRpartPlot(fit) 



#####################################################################################################
################## Common Functions  #################
#####################################################################################################


# Common Functions including the backward selection for the first Question

set.seed(24287)


bindModel <- function(yLabel, xFeatures){
  # Automates the creation of feature model to be passed into an Classifier or Predictive Model  
  return (as.formula(paste(yLabel, "~", paste(xFeatures, collapse = '+ '))))
}


# Takes the complete dataframe as an input including the label column
factorToDummy_DF_Builder <- function (dataFrameIN, numericCols, factorCols, labelCol){
  # Creates a design matrix by expanding the factors to a set of dummy variables and interaction etc.
  xNumeric <- dataFrameIN[, numericCols]
  xFactor <- dataFrameIN[, c(factorCols,labelCol)]
  
  factorModel <- bindModel(yLabel=labelCol, xFeatures=factorCols)
  xFactor <- model.matrix(factorModel, data=xFactor)[, -1]        # -1 is provided to exclude the intercept term from the matrix
  yLabel <- dataFrameIN[labelCol]
  return (data.frame(xNumeric, xFactor, yLabel))
}


stratifiedSampling <- function(dataIN, sample_on_col, trainPrcnt){
  trainIndices <- createDataPartition(y=dataIN[[sample_on_col]], p=trainPrcnt, list=FALSE)
  trainData <- dataIN[trainIndices,]
  testData <- dataIN[-trainIndices,]
  
  stopifnot(nrow(trainData) + nrow(testData) == nrow(dataIN))
  return (list(trainData, testData))
}


# Plot and calculate the acuracy, precision and recall for different range of cut-offs
# For a credit default we are more interested in having a high recall or high recall
performanceMetric <- function (cutoffRange, y, y_hat){
  y_bin <- y_hat
  actualYesIndex <- which(y==1)
  #     perfMetric <- data.frame()
  perfMetric <- matrix(0,length(cutoffRange),3)    # 3 is because we calculate accuracy, recall and precision
  for (i in 1:length(cutoffRange)){
    #         print (cutOFF)
    predYesIndex <- which(y_hat>=cutoffRange[i])
    bothYesIndex <- intersect(actualYesIndex,predYesIndex)
    
    # Get the Binomial prediction based on cut-off value
    y_bin[predYesIndex] <- 1
    y_bin[-predYesIndex] <- 0
    
    # Calculate the accuracy, precision and recall
    accuracy <- length(which(y_bin == y))/length(y)
    precision <- length(bothYesIndex)/length(predYesIndex)
    recall <- length(bothYesIndex)/length(actualYesIndex)
    cbind(accuracy, precision, recall)
    
    perfMetric[i,] <- cbind(accuracy, precision, recall)
  }
  
  return (perfMetric)
  
}


# Changing the datatypes
changeDataType <- function(dataIN, featureNames, type){
  if (type=='factor'){
    dataIN[featureNames] <- lapply(dataIN[featureNames], factor)
  }
  else if (type=='numeric'){
    dataIN[featureNames] <- lapply(dataIN[featureNames], as.numeric)
  }
  else{
    print ('No Type Specified! Specify a Type Factor or Numeric')
  }
  return (dataIN)
}


aicCompute <- function(fullModel, dataIN){
  glmIN <- glm(fullModel, data = dataIN)
  aic <- AIC(glmIN)
  return (aic) 
}



backwardSelection <- function(features, label, dataIN){
  featuresIN <- features
  while (TRUE){
    fullModel <- bindModel(label, featuresIN)
    aic_main <- aicCompute(fullModel, dataIN)
    #         print ('AIC Main')
    #         print (aic_main)
    intermediateAIC <- c()
    for (j in (1:length(featuresIN))){
      newFeatureSet <- featuresIN[-j]
      newModel <- bindModel(label, newFeatureSet)
      aicNew <- aicCompute(newModel, dataIN)
      intermediateAIC <- c(intermediateAIC, aicNew)
    }
    #         print ('AIC List')
    #         print (intermediateAIC)
    
    badFeatureIndex <- which(intermediateAIC == min(intermediateAIC))
    featuresIN <- featuresIN[-badFeatureIndex]
    
    #         print (fullModel)
    if (aic_main < min(intermediateAIC)){
      return (fullModel)
    }
  }
}


plotPerfMetric <- function(performanceDF, cutoffRange){
  p <- ggplot() + 
    geom_line(data = performanceDF, aes(x = cutoffRange, y = accuracy, color = "accuracy")) +
    geom_line(data = performanceDF, aes(x = cutoffRange, y = precision, color = "precision")) +
    geom_line(data = performanceDF, aes(x = cutoffRange, y = recall, color = "recall")) +
    xlab('Cutoff') +
    ylab('percent.change')
  return (p)
}



#####################################################################################################
#####################################################################################################


# 2. Download the credit card default data set from the UCI machine learning
# repository. https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

dataDir <- "/Users/sam/All-Program/App-DataSet/Data-Science-Projects/Credit-Default/default_of_credit_card_clients.xls"
data <- read.xls (dataDir, sheet = 1, skip=1, header = TRUE)
head(data)
# data1<-read.xls(file="default_of_credit_card_clients.xlsx", sheet = 1, skip = 1, header = TRUE)

# Remove the ID column:
credit.data <- subset(data, select=-c(ID))
head(credit.data)
dim(credit.data)

# Change the label column name
colnames(credit.data)[24] <- "default"




#####################################################################################################
#####################################################################################################



# 3. Identify all the relevant categorical, numeric, and logical variables.

# Findings:
# It seems that [sex, education, marriage, age] are nominal
# All other variales are either numerical
# There's is only one logical variable "default" as it can take only two value either "Yes (True)" or "No (False)"
# 
# PAY_0 : September,
# PAY_2 : August
# PAY_3 : July
# PAY_4 : June
# PAY_5 : May
# PAY_6 : April
# -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above. 

uniqueCount <- function (feature){
  return (length(unlist(unique(credit.data[feature]))))
} 
sapply(colnames(credit.data), FUN=uniqueCount)



numericCols <- names(which(sapply(credit.data, is.numeric)))
nominalCols <- names(which(sapply(credit.data, is.factor)))
print (nrow(credit.data))
print (ncol(credit.data))

# Convert into Proper datatypes
credit.nominalCols <- c('SEX','EDUCATION','MARRIAGE')
credit.numericCols <- setdiff(colnames(credit.data), credit.nominalCols)
credit.data <- changeDataType(credit.data, credit.nominalCols, type='factor')
credit.data <- changeDataType(credit.data, credit.numericCols, type='numeric')


# CAPTURE Numeric and nominal and the label Columns
credit.labelCol <- 'default'
credit.numericCols <- setdiff(names(which(sapply(credit.data, is.numeric))), credit.labelCol)
credit.nominalCols <- names(which(sapply(credit.data, is.factor)))

str(credit.data)
credit.labelCol
credit.numericCols
credit.nominalCols

# # Check if data is missing
# There is no missing Data



#####################################################################################################
#####################################################################################################


# 4. Perform all required EDA on this data set.

# ScatterPlot
# We select a small sample to plot
# plot(credit.data)

set.seed(24287)
samplePrcntg <- 0.10
credit.sampleIndices <- createDataPartition(y = credit.data$default, p=samplePrcntg, list=FALSE)
credit.sample <- credit.data[credit.sampleIndices , ]

head(credit.sample)
nrow(credit.sample)

# Correlation Matrix for numerical data
  options(repr.plot.width=15, repr.plot.height=10)
  credit.cor_matrix <- cor(credit.data[, c(credit.numericCols, credit.labelCol)]) # Building the correlation plot
  corrplot(credit.cor_matrix, method="number") 
  
  
# Scatter Plots for numerical attributes
  options(repr.plot.width=15, repr.plot.height=20)
  plot(credit.sample[, credit.numericCols])
  
  
# Box Plots for numerical attributes
  # dev.off()
  options(repr.plot.width=10, repr.plot.height=15)
  par(mfrow=c(4,5))
  crearteBoxPlots <- function (column_name, dataIN){
    #     print (column_name)
    #     print (nrow(dataIN))
    boxplot(dataIN[column_name], horizontal = FALSE,  main= column_name)
    #     stripchart(dataIN[column_name], add = TRUE, pch = 20, col = 'red')
  } 
  a <- sapply(credit.numericCols, FUN=crearteBoxPlots, dataIN=credit.sample)
  
  
# Histograms for all numerical attributes
  options(repr.plot.width=10, repr.plot.height=10)
  ggplot(data = melt(credit.data[, credit.numericCols]), mapping = aes(x = value)) + 
    geom_histogram(bins = 10) + facet_wrap(~variable, scales = 'free_x')
  
  
# Bar distribution plots for nominal attributes
  options(repr.plot.width=10, repr.plot.height=5)
  par(mfrow=c(1,3))
  barPlots <- function(featureVector, dataIN){
    tab <- table(dataIN[featureVector])
    #     print (tab)
    barplot(tab, main=featureVector, xlab="Feature Categories")
  }
  sapply(credit.nominalCols, FUN=barPlots, credit.data)
  
  
# Bar distribution plots for of class variable grouped on nominal variables
  options(repr.plot.width=10, repr.plot.height=10)
  par(mfrow=c(3,1))
  crossTab_barplots <- function(featureVector, dataIN, labelCol){
    tab <- table(dataIN[[featureVector]], dataIN[[labelCol]]) 
    barplot(tab, main=featureVector,
            xlab=labelCol,
            legend = rownames(tab), beside=TRUE)
  }
  
  sapply(credit.nominalCols, FUN=crossTab_barplots, credit.data, 'default')
  
  
  
################################################
############    STRATIFIED SAMPLING  ###########
################################################


# Standarize the dataset with 0 mean and unit variance
  credit.data.scaledNumeric <- scale(credit.data[credit.numericCols])
  
  # Check if the mean is 0 and is unit variance
  stopifnot(colMeans(credit.data.scaledNumeric) != 0)
  stopifnot(round(apply(credit.data.scaledNumeric, 2, sd)) == 1)
  
  credit.data.scaled <- cbind(credit.data[credit.nominalCols], credit.data.scaledNumeric, credit.data['default'])
  
  head(credit.data)
  head(credit.data.scaled)
  

  
# Stratified sampling on original (non-standarize) data
  # Get the Null model and the Full model
  credit.dataIN <- credit.data
  credit.null.model <- as.formula(paste('default', "~", 1))
  credit.full.model <- bindModel(yLabel = 'default',xFeatures = c(credit.nominalCols, credit.numericCols))
  
  credit.null.model
  credit.full.model
  
  # Get the Train Test Data
  dataOUT <- stratifiedSampling(dataIN=credit.dataIN, sample_on_col='default', trainPrcnt = 0.8)
  
  credit.trainData <- dataOUT[[1]]
  credit.testData <- dataOUT[[2]]
  nrow(credit.trainData)
  nrow(credit.testData)
  head(credit.trainData)
  

  
# Stratified sampling on standarize data
  # Get the Null model and the Full model
  credit.dataIN <- credit.data.scaled
  credit.null.model <- as.formula(paste('default', "~", 1))
  credit.full.model <- bindModel(yLabel = 'default',xFeatures = c(credit.nominalCols, credit.numericCols))
  
  credit.null.model
  credit.full.model
  
  # Get the Train Test Data
  dataOUT <- stratifiedSampling(dataIN=credit.dataIN, sample_on_col='default', trainPrcnt = 0.8)
  
  credit.trainData.sc <- dataOUT[[1]]
  credit.testData.sc <- dataOUT[[2]]
  nrow(credit.trainData.sc)
  nrow(credit.testData.sc)
  head(credit.trainData.sc)
  
  
  
# Train Test for dummy expanded matrix
  credit.data.dummy <- factorToDummy_DF_Builder(dataFrameIN = credit.data.scaled, 
                                                numericCols = credit.numericCols, 
                                                factorCols = credit.nominalCols,
                                                labelCol = credit.labelCol)
  
  # credit.data.dummy
  dataOUT <- stratifiedSampling(dataIN = credit.data.dummy, sample_on_col = credit.labelCol, trainPrcnt = 0.8)
  credit.trainData.dummy <- dataOUT[[1]]
  credit.testData.dummy  <- dataOUT[[2]]
  
  nrow(credit.trainData.dummy)
  nrow(credit.testData.dummy)
  head(credit.testData.dummy)
  
  
  
  
#####################################################################################################
#####################################################################################################

# 5.Build a logistic regression model to determine whether or not a
# customer defaulted. Use all of the variables. Validate the model on a
# test data set. Use the comments to discuss the performance of the model.


# Findings:  We see that the model is not a very good model as the highest accuracy (at threshold 0.43) we observe is about 80% and at that threshold the precision and recall of finding credit default is very less. For our model we would be interested in catching as many credit default as possible. The model howver at a threshold of about 0.27 has both recall and precision at about 52% and the accuracy at about 77%.
  
  # Fit The Null Model
  credit.glm.null <- glm(formula=credit.null.model, family=binomial(logit), data=credit.trainData.sc)
  summary(credit.glm.null)
  
  # Fit The Full Model
  credit.glm.full <- glm(formula=credit.full.model, family=binomial(logit), data=credit.trainData.sc)
  # credit.glm.full <- glm(formula=newModel, family=binomial(logit), data=credit.trainData.sc)
  summary(credit.glm.full)
  
  # Predict for the Full model
  credit.testData.sc$defaultPred <- predict(credit.glm.full, newdata=credit.testData.sc, type="response")
  
 
  # Evaluation: (Performance metric - accuracy , precison and recall)
  # Range for cuttoff
  cutoffRange <- seq(.01,.99,length=1000)
  perfMatrix <- performanceMetric(cutoffRange, credit.testData.sc$default, credit.testData.sc$defaultPred)
  perfDF <- data.frame(perfMatrix)
  names(perfDF) <- c('accuracy', 'precision', 'recall')
  head(perfDF)
  
  # Plot Accuracy, precision and recall
  options(repr.plot.width=6, repr.plot.height=4)
  p <- plotPerfMetric(perfDF, cutoffRange)
  p
  
  
  
#####################################################################################################
#####################################################################################################
  
# 6. Using forward selection, determine the best model.

# Findings:
# The best model using forward selection is: 
#   
# default ~ PAY_0 + LIMIT_BAL + PAY_3 + PAY_AMT1 + MARRIAGE + BILL_AMT1 + 
#    EDUCATION + PAY_AMT2 + BILL_AMT2 + PAY_2 + SEX + PAY_AMT4 
# 
# The prediction using the full model and the best model using the forward selection technique are pretty close, thats the reason we dont see any improvement in the performance metric using the forward selected best model.

  
  credit.glm.forward = step(credit.glm.null,scope=list(lower=credit.null.model,upper=formula(credit.full.model)), direction="forward")
  credit.glm.fowardbestModel <- formula(credit.glm.forward)
  
  credit.glm.full.forward <- glm(formula=credit.glm.fowardbestModel, family=binomial(logit), data=credit.trainData.sc)
  
  # Predict for the Full model
  credit.testData.sc$defaultPredForward <- predict(credit.glm.full.forward, 
                                                   newdata=credit.testData.sc, 
                                                   type="response")
  
  
  # Model Evaluation;
  
  # Range for cuttoff
  cutoffRange <- seq(.01,.99,length=1000)
  perfMatrix <- performanceMetric(cutoffRange, credit.testData.sc$default, credit.testData.sc$defaultPredForward)
  perfDF <- data.frame(perfMatrix)
  names(perfDF) <- c('accuracy', 'precision', 'recall')
  head(perfDF)
  
  # Plot Accuracy, precision and recall
  options(repr.plot.width=6, repr.plot.height=4)
  p <- plotPerfMetric(perfDF, cutoffRange)
  p
  
  
  

#####################################################################################################
#####################################################################################################
  
  
# 7. Using the backwards selection function you implemented in #1
# , determine the best model.

# Best Model:
# default ~ SEX + EDUCATION + MARRIAGE + LIMIT_BAL + AGE + PAY_0 + 
#   PAY_2 + PAY_3 + PAY_5 + BILL_AMT1 + BILL_AMT2 + PAY_AMT1 + 
#   PAY_AMT2 + PAY_AMT4 + PAY_AMT5
  
  # Backward Selection:
  allFeatures <- c(credit.nominalCols, credit.numericCols)
  print (length(allFeatures))
  
  bestModel <- backwardSelection(features=allFeatures, label=credit.labelCol, dataIN=credit.data)
  bestModel
  
  credit.glm.backward.manual <- glm(formula=bestModel, family=binomial(logit), data=credit.trainData.sc)
  credit.testData.sc$defaultPredBackward_Manual <- predict(credit.glm.backward.manual, newdata=credit.testData.sc, type="response")
  
  
  # Evaluation (Precison, recall and accuracy):
  # Range for cuttoff
  cutoffRange <- seq(.01,.99,length=1000)
  perfMatrix <- performanceMetric(cutoffRange, credit.testData.sc$default, credit.testData.sc$defaultPredBackward_Manual)
  perfDF <- data.frame(perfMatrix)
  names(perfDF) <- c('accuracy', 'precision', 'recall')
  head(perfDF)
  
  # Plot Accuracy, precision and recall
  options(repr.plot.width=6, repr.plot.height=4)
  p <- plotPerfMetric(perfDF, cutoffRange)
  p
  
  
  
  
#####################################################################################################
#####################################################################################################
  

# 8. Run an implementation of backwards selection found in an R package on this
# data set. Discuss any differences between the results of this implementation
# and your implemnetation in question 7.

# Best Model:
# default ~ SEX + EDUCATION + MARRIAGE + LIMIT_BAL + AGE + PAY_0 + 
#   PAY_2 + PAY_3 + PAY_5 + BILL_AMT1 + BILL_AMT2 + BILL_AMT5 + 
#   PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5

# Findings:
# The manual backward selection implementation seems like more parsimonious as the featrure space is less compared to the backward selection implemented by R Package. Features [BILL_AMT5, PAY_AMT5] are said to be singificant in the backward selection implemented by R package, whereas these were not identified as significant by the manual backward selection model. 

  credit.glm.backward <- step(credit.glm.full)
  credit.glm.backwardbestModel <- formula(credit.glm.backward)
  
  credit.glm.full.backward <- glm(formula=credit.glm.backwardbestModel, family=binomial(logit), data=credit.trainData.sc)
  summary(credit.glm.full.backward)
  
  # Predict for the Full model
  credit.testData.sc$defaultPredBackward <- predict(credit.glm.full.backward, newdata=credit.testData.sc, type="response")
  
  # --> Evalution GLM best model with Backward Elimination 
  # Range for cuttoff
  cutoffRange <- seq(.01,.99,length=1000)
  perfMatrix <- performanceMetric(cutoffRange, credit.testData.sc$default, credit.testData.sc$defaultPredBackward)
  perfDF <- data.frame(perfMatrix)
  names(perfDF) <- c('accuracy', 'precision', 'recall')
  head(perfDF)
  
  # Plot Accuracy, precision and recall
  options(repr.plot.width=6, repr.plot.height=4)
  p <- plotPerfMetric(perfDF, cutoffRange)
  p
  
  
  
#####################################################################################################
#####################################################################################################  
  
  
  
# 9. Run lasso regression on the data set. Briefly discuss how you determined
# the appropriate tuning parameters.

# Findings: We run Lasso using the GLMNET package cross validation. Lasso in GLMNET can be run by using alpha =1 as an input to GLMNET function. The cv.glmnet fits the model for nfold crossvalidation and evaluates the model for each crossvalidation for different values of lamda (l1 norm). The value of lambda that gives the best performance metric during each cross-validation is chosen as the best lambda. This value of the lambda can be determined by "lambda.min". The prediction on the fitted model can simply be done by calling the predict function.
  
  # Split the label from the Train and Test Data
  xTrainData <- credit.trainData.dummy[, -which(names(credit.trainData.dummy) == credit.labelCol)]
  yTrainLabel <- credit.trainData.dummy[credit.labelCol]
  xTestData <- credit.testData.dummy[, -which(names(credit.testData.dummy) == credit.labelCol)]
  yTestLabel <- credit.testData.dummy[credit.labelCol]
  
  credit.lasso.cv = cv.glmnet(x=as.matrix(xTrainData), y=as.matrix(yTrainLabel), alpha=1, family='binomial')
  credit.lasso.predict <- predict(credit.lasso.cv, newx = as.matrix(xTestData), s = "lambda.min", type = "response")
  options(repr.plot.width=10, repr.plot.height=4)
  par(mfrow=c(1,2))
  plot(credit.lasso.cv, main="LASSO")
  
  # Range for cuttoff
  cutoffRange <- seq(.01,.99,length=1000)
  perfMatrix <- performanceMetric(cutoffRange = cutoffRange, 
                                  y = yTestLabel$default, 
                                  y_hat = unlist(credit.lasso.predict))
  
  perfDF <- data.frame(perfMatrix)
  names(perfDF) <- c('accuracy', 'precision', 'recall')
  head(perfDF)
  
  # Plot Accuracy, precision and recall
  options(repr.plot.width=6, repr.plot.height=4)
  p <- plotPerfMetric(perfDF, cutoffRange)
  p
  

  
  
#####################################################################################################
##################################################################################################### 
  
  
# 10. Run ridge regression on the data set. Briefly discuss how you determined
# the appropriate tuning parameters.

# Findings: 
#   First 
# 
# We run RIDGE using the GLMNET package cross validation. RIDGE in GLMNET can be run by using alpha =0 as an input to GLMNET function. 
# 
# We do cross validation using the GLMNET package. cv.glmnet fits the model for nfold crossvalidation and evaluates the model for each crossvalidation for different values of lamda (l2 norm). The value of lambda that gives the best performance metric during each cross-validation is chosen as the best lambda. This value of the lambda can be determined by "lambda.min" which was found to be 0.01469156. The prediction on the fitted model can simply be done by calling the predict function.
  

  # Here we use glmnet package to do ridge regression
  # Find the best lambda and predict on that lambda for the test set.
  credit.ridge.cv <- cv.glmnet(x=as.matrix(xTrainData), y=as.matrix(yTrainLabel), alpha=0, family='binomial')
  lambdaBest <- credit.ridge.cv$lambda.min
  credit.ridge.fit <- glmnet(x=as.matrix(xTrainData), y=as.matrix(yTrainLabel), alpha=0, lambda=credit.ridge.cv$lambda.min, family='binomial')
  credit.ridge.predict <- predict(credit.ridge.fit, newx = as.matrix(xTestData), s = lambdaBest, type = "response") 
  
  options(repr.plot.width=10, repr.plot.height=4)
  par(mfrow=c(1,2))
  plot(credit.ridge.cv, main="RIDGE")
  
  
  # Range for cuttoff
  cutoffRange <- seq(.01,.99,length=1000)
  perfMatrix <- performanceMetric(cutoffRange = cutoffRange, 
                                  y = yTestLabel$default, 
                                  y_hat = unlist(credit.ridge.predict))
  
  perfDF <- data.frame(perfMatrix)
  names(perfDF) <- c('accuracy', 'precision', 'recall')
  head(perfDF)
  
  # Plot Accuracy, precision and recall
  options(repr.plot.width=6, repr.plot.height=4)
  p <- plotPerfMetric(perfDF, cutoffRange)
  p

  
  # credit.ridge.full <- lm.ridge(formula=credit.full.model, data=credit.data.scaled, lambda = seq(-50,50,0.01))
  # select(credit.ridge.full)   # We see that the best lambda is at 35.24.
  
#####################################################################################################
##################################################################################################### 


# 11. Run naive bayes on the data set.

  # credit.full.model
  credit.nb.fit <- naiveBayes(credit.full.model, data = credit.trainData)
  credit.nb.predict <- predict(credit.nb.fit, newdata = credit.testData, type = 'raw')  
  
  preds <- (credit.nb.predict[,'0'] <= credit.nb.predict[,'1'])*1
  # conf_matrix <- table(preds, yTestLabel$default)
  confusionMatrix(reference = yTestLabel$default, data = preds, positive = "1", mode='prec_recall')
  
  # Range for cuttoff
  cutoffRange <- seq(.01,.99,length=1000)
  perfMatrix <- performanceMetric(cutoffRange = cutoffRange, 
                                  y = yTestLabel$default, 
                                  y_hat = credit.nb.predict[,2])
  
  perfDF <- data.frame(perfMatrix)
  names(perfDF) <- c('accuracy', 'precision', 'recall')
  head(perfDF)
  
  # Plot Accuracy, precision and recall
  options(repr.plot.width=6, repr.plot.height=4)
  p <- plotPerfMetric(perfDF, cutoffRange)
  p


  
#####################################################################################################
##################################################################################################### 


# 12. Build a decision tree to classify the customers as defaulted
# or not-defaulted. Plot the resulting tree. Discuss whether you
# feel this is a good model.
  
# Findings: In terms of Accuracy (82.08%), the model seems reasonable to some extent. However, the dataset is inbalanced in terms of class and there are reletively more instances for non defaults than there are for defaults. Hence for this scenario we need to see the precision and recall (for the "default" case) exhibited by the model on the test data. As we see that the precision (69.6) is reasonable compared to other models but the recall (33.635) is very less, we say that the simple decision tree model is not a good model. 
# Moreover, given that the level of the tree is only 1, which means only 1 featurs define the data, The model is too small but exibit comparable accuracy and precision when comapared to other model. The model is a reasonally good model but not the best as the risk of identifying less defaults is more than identifying non-defaults.
  
  credit.dt.fit <- rpart(credit.full.model, data=credit.trainData, method="class")
  credit.dt.predict <- predict(credit.dt.fit, credit.testData, type = "class")
  credit.testData$defaultPredDT <- credit.dt.predict  

  CM <- confusionMatrix(reference = credit.testData$default, data = credit.testData$defaultPredDT, positive = "1", mode='prec_recall')

  CM <- confusionMatrix(reference = credit.testData$default, 
                        data = credit.testData$defaultPredDT, 
                        positive = "1", mode='prec_recall')
  CM
  
  
  library(rpart.plot)
  library(rattle)
  fancyRpartPlot(credit.dt.fit)
  
  
  

  
#####################################################################################################
##################################################################################################### 
  
  
# 13. Build a random forest model and apply it to classify the test data set.

  # Note Very IMPORTANT !!!!! 
  # Convert the class label into factor,
  # If you dont do so, random forest is gonna
  # treat it as a regression and throw a warning 
  # that the response variable has very few unique values
  
  x <- subset(credit.trainData, select=-c(default))
  y <- as.factor(as.character(credit.trainData$default))
  
  
  credit.rf.fit <- randomForest(x = x,
                                y = y,
                                importance = TRUE,
                                #method = "class",      # Choose what type of prediction you wanna make.
                                ntree = 200)
  
  credit.rf.predict <- predict(credit.rf.fit, credit.testData, type = "response")
  
  credit.testData$defaultPredRF1 <- credit.rf.predict
  
  
  # Model Evaluation
  confusionMatrix(reference = credit.testData$default, 
                  data = credit.testData$defaultPredRF1, 
                  positive = "1", 
                  mode='prec_recall')
  
  
  
  
#####################################################################################################
##################################################################################################### 
  
  
  
# 14. Discuss the comparative performance of all of the models used. How should
# we determine which is best? Provide justification for your answer.
  
# Findings  
  # In the case of identifying default, it is important that we have high recall and an overall accuracy. Most of the models such as Logistic Regression (RIDGE/LASSO), Forward selection and Backward selection gives the best model performance at threshold approximately 0.27. The accuracy is seen as approximately 77% with precision and recall at 51%. Naive Bayes model perform very poorly compared to all other models and it makes sense because Naive bayes assumes that the features are independent given the response variable. But in the real world the features are often dependent on each other. The decision tree model and random forest are very close in terms of accuracy precision and recall. The Random Forest model exhibits a accuracy of 82.47, precision of 68.4 and recall of 38.3. Decision tree model has better precision (only by 1%) and the random forest model has more reecall (by 5%). 
  
  # Domain expertise would be valuable in this case as models produce different results. In case of high accuracy and precision all the models are a good fit. As Lasso/Ridge and BS/FS GLM all exhibit reasonabaly well accuracy and precesion at cut-off 0.4. However given the domain problem Recall is considered to be of high importance. Hence, on average Lasso, Ridge, GLM with feature selection with properly tuning the cut-off at [0.27-0.30] range would be more flexible to use and provides reasonable accuracy, precision and recall. And for all these models we have vey similary accuracy, precision and recall for nearby cut-off value.