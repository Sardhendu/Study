# CSP571
# Homework 3
# Clean all the vatriable from the worlspace
rm(list = ls())



############################################################################################################################################################################################################################


# 1.Load in the auto mpg data set: https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data

  url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
  datum <- read.table(url, header=FALSE)
  autoMPG.datum <- data.frame(datum)
  head(autoMPG.datum)



############################################################################################################################################################################################################################



# 2. Identify all of the categorical variables, all of the numeric variables
# and all of the binary variables.

# Doubt: Are there any Binary Variables?, when represented as factors, can the categorial columns be told as binary variables?
  
  # Adding relevant feature names to the dataset
  featureNames <- c("mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name")
  names(autoMPG.datum) <- featureNames
  # Identifying using the data fetched from the source.
  autoMPG.orig_numericCols <- names(which(sapply(autoMPG.datum, is.numeric)))
  autoMPG.orig_numericCols
  autoMPG.orig_factorCols <- names(which(sapply(autoMPG.datum, is.factor)))
  autoMPG.orig_factorCols
  
  # Identifying Correct datatypes for the variables using unique, The Funda is that the columns thar have relatively less than 10 unique categories can be justified as factor columns.
  uniqueCount <- function (feature){
    print (length(unlist(unique(autoMPG.datum[feature]))))
  } 
  sapply(featureNames, FUN=uniqueCount)
  
  # Using the unique count it seems like cylinders, model_year and origin are discrete varables and all other features are numeric variables. However there seems to be no binary variable.
  
  # 1. mpg:           continuous
  # 2. cylinders:     multi-valued discrete
  # 3. displacement:  continuous
  # 4. horsepower:    continuous
  # 5. weight:        continuous
  # 6. acceleration:  continuous
  # 7. model year:    multi-valued discrete
  # 8. origin:        multi-valued discrete
  # 9. car name:      string (unique for each instance)
  
  # Converting all the numeric columns into numeric and nonnumeric columns into factors
  autoMPG.datumVarDtype <- autoMPG.datum
  autoMPG.datumVarDtype$mpg <- as.numeric(as.character(autoMPG.datum$mpg))
  autoMPG.datumVarDtype$displacement <- as.numeric(as.character(autoMPG.datum$displacement))
  autoMPG.datumVarDtype$horsepower <- as.numeric(as.character(autoMPG.datum$horsepower))
  autoMPG.datumVarDtype$weight <- as.numeric(as.character(autoMPG.datum$weight))
  autoMPG.datumVarDtype$acceleration <- as.numeric(as.character(autoMPG.datum$acceleration))
  
  autoMPG.datumVarDtype$cylinders <- as.factor(as.character(autoMPG.datum$cylinders))
  autoMPG.datumVarDtype$model_year <- as.factor(as.character(autoMPG.datum$model_year))
  autoMPG.datumVarDtype$origin <- as.factor(as.character(autoMPG.datum$origin))
  head(autoMPG.datumVarDtype)
  
  # Remove the last column (car_name) from teh dataset as the car name is a primary key and hence should not be included in the model.
  autoMPG.datumClean <- autoMPG.datumVarDtype
  autoMPG.datumClean$car_name <- NULL
  head(autoMPG.datumClean)
  
  sapply(autoMPG.datumClean, class)
  
  autoMPG.datumCleanAlias <- autoMPG.datumClean




############################################################################################################################################################################################################################





# 3. Identify the appropriate descriptive statistics and graph for this data set.
# Execute on those and use the comments to discuss relevant relationships or insights discovered.

# Doubt: how to plot equivallent of plot in ggplot with smooth line?
# Doubt: Is there a way to do barplots (using subplots) using the sql like tables of discrete colums given the count of the categorical values. 
  
# Things to do for analysis
# Density plot, Histogram plot, Box plot, 
# Use the signed log transformation if the data is skewed
# Check if the data has high variance using histograms if the range is too large
# Check if the mean and median are close by
# Check if the data is equally distributed for categorical labels.
  
  
  library(ggplot2)
  library(grid)
  library(gridExtra)
  
  autoMPG.numericCols <- c("mpg", "displacement", "horsepower", "weight", "acceleration")
  autoMPG.discreteCols <- c("cylinders", "model_year", "origin")
  
  # 1. General summary
      summary(autoMPG.datumClean)
  
  
  # 2. Box plots for numerical columns
      dev.off() 
      par(mfrow=c(2,3))
      autoMPG.crearteBoxPlots <- function (column_name, dataIN){
        boxplot(dataIN[column_name], horizontal = TRUE,  main= column_name)
        stripchart(dataIN[column_name], add = TRUE, pch = 20, col = 'red')
      } 
      sapply(autoMPG.numericCols, FUN=autoMPG.crearteBoxPlots, dataIN=autoMPG.datumClean)
  
  
  # 3. Scatter plot with smooth line fit for numerical columns. 
      dev.off()
      plot(autoMPG.datumClean[, autoMPG.numericCols])
      
      
  # 4. Plotting Density Histograms for numerical columns
      library(reshape2)
      dev.off() 
      ggplot(data = melt(autoMPG.datumClean), mapping = aes(x = value)) + 
        geom_histogram(bins = 10) + facet_wrap(~variable, scales = 'free_x')

  
  # 5. Distribution of counts of categories for discrete variables. 
      datumClean_alias <- autoMPG.datumClean[, autoMPG.discreteCols]
      
      discreteColTable <- sqldf("select 'cylinders' as feature, cylinders as category, count(cylinders) as count from datumClean_alias group by cylinders union select 'model_year' as feature, model_year as category, count(model_year) as count from datumClean_alias group by model_year union select 'origin' as feature, origin as category, count(origin) as count from datumClean_alias group by origin")
      
      
  # 6. Showcasing the above as Barplots:
      dev.off()
      par(mfrow=c(2,2))
      barPlots <- function(featureVector, dataIN){
        tab <- table(dataIN[featureVector])
        barplot(tab, main=featureVector, xlab="Feature Categories")
      }
      sapply(autoMPG.discreteCols, FUN=barPlots, autoMPG.datumClean)
      
  
  # 7. Box plot for each continuous variable given each categorical value of categorical features. This plot would be benefitial to understand the distribition of the continuos variable given a categorival value of a discrete variable
  
      # For discrete variable "cylinder"
          dev.off() 
          par(mfrow=c(3,2))
          levels(autoMPG.datumClean$cylinders)
          autoMPG.cylinder3 <- autoMPG.datumClean[which(autoMPG.datumClean$cylinders == 3), ]
          sapply(autoMPG.numericCols, FUN=autoMPG.crearteBoxPlots, dataIN=autoMPG.cylinder3)
          
          dev.off() 
          par(mfrow=c(3,2))
          autoMPG.cylinder4 <- autoMPG.datumClean[which(autoMPG.datumClean$cylinders == 4), ]
          sapply(autoMPG.numericCols, FUN=autoMPG.crearteBoxPlots, dataIN=autoMPG.cylinder4)
          # mpg -> one outlier to investigate
          # acceleration -> four outliers to investigate
          
          dev.off() 
          par(mfrow=c(3,2))
          autoMPG.cylinder5 <- autoMPG.datumClean[which(autoMPG.datumClean$cylinders == 5), ]
          sapply(autoMPG.numericCols, FUN=autoMPG.crearteBoxPlots, dataIN=autoMPG.cylinder5)
          
          dev.off() 
          par(mfrow=c(2,3))
          autoMPG.cylinder6 <- autoMPG.datumClean[which(autoMPG.datumClean$cylinders == 6), ]
          sapply(autoMPG.numericCols, FUN=autoMPG.crearteBoxPlots, dataIN=autoMPG.cylinder6)
          # mpg -> 5 outliers to investigate
          # horsepower -> One extreme outlier to investigate
          
          dev.off() 
          par(mfrow=c(3,2))
          autoMPG.cylinder8 <- autoMPG.datumClean[which(autoMPG.datumClean$cylinders == 8), ]
          sapply(autoMPG.numericCols, FUN=autoMPG.crearteBoxPlots, dataIN=autoMPG.cylinder8)
          
      # For discrete variable "origin" 
          # origin = 1 “North America”, 
          # origin = 2 “Europe”, 
          # origin = 3 “Asia”
          
          dev.off() 
          par(mfrow=c(2,3))
          levels(autoMPG.datumClean$origin)
          autoMPG.origin1 <- autoMPG.datumClean[which(autoMPG.datumClean$origin == 1), ]
          sapply(autoMPG.numericCols, FUN=autoMPG.crearteBoxPlots, dataIN=autoMPG.origin1)
          # mpg -> 2 outliers
          
          dev.off() 
          par(mfrow=c(2,3))
          levels(autoMPG.datumClean$origin)
          autoMPG.origin2 <- autoMPG.datumClean[which(autoMPG.datumClean$origin == 2), ]
          sapply(autoMPG.numericCols, FUN=autoMPG.crearteBoxPlots, dataIN=autoMPG.origin2)
          # mpg -> 6 extreme outliers
          # displacement -> 2 outliers , one extreme
          # horsepower -> 2 outliers, one extreme
          
          dev.off() 
          par(mfrow=c(2,3))
          levels(autoMPG.datumClean$origin)
          autoMPG.origin3 <- autoMPG.datumClean[which(autoMPG.datumClean$origin == 3), ]
          sapply(autoMPG.numericCols, FUN=autoMPG.crearteBoxPlots, dataIN=autoMPG.origin3)
          # mpg -> 1 outlier
  
  # Findings For Discrete variables:
      # It can be easily seen from the Barplots that the distibution of the feature variable "Cylinder" is not Uniform. For Example their are more data on "4 cylinders" and very less data on "3 cylinders" and "5 Cylinders" this could result is skewness of some variables that are highly correlated to the cylinder feature. Similarly, the feature "Origin" also has non-uniform distribution. How having non-uniform distribution affects the model and other feature variable are given below.
      
  # Relationships: 
      # Almost all the numerical attributes are somewhat -vely or +vely related to each other) for example
      # Greater horsepower = more cubic centiliter (more cylinders and displacement) --> more weight --> less MPG
      # For example : 
      # For 4 Cylinder: median horsepower = 78, median weight = 2200, median mpg = 27, median displacement = 105
      # For 8 Cylinder: median horsepower = 150, median weight = 4200, median mpg = 14, median displacement = 350
      
  # For mpg:
    # Contains 1 outlier
    # A little bit right skewed (right whisker is longer than left), but the Mean and median are very close, the little skewness might be the result of the outlier or very few data at the right end that do not overly affect the mean.

  # For cylinders:
    # The feature is highly biased towards the 4, and 8 cylinders and a little towards 6th cylinder. The category 3 and 5 cylinders have very less corresponding data points
  
  # For Displacement & Weight & horsepower
    # Horse power:Heavily right skewed -> With a median 151 and mean 194.
    # Weight: Right Skewed -> Median 2804 mean =2978
    # horsepower : Right skewed -> median=93.5 mean= 104.5 
    # The major reason for the skewness is because majority of the dataset is biased towards "4 cylinders". Since half of the dataset belongs to "4 cylinders", its reasonable to assume that a 4 cylinder car would have lower horsepower, weight and displacement compared to 6 cylinders and 8 cyilnders. 
    # Perfoming further analysis on the subset of data with "4 cylinders" we see the below outputs
          # 4 cylinders: median displacement = 105 much closer to 151
          # 4 cylinders: median horsepower = 80 much closer to 93
          # 4 cylinders: median weight = 2200 reasonably closer to 2804
          
  
  # HorsePower 
    # The horsepower variale has few mising value (6)
    # contains 6 outliers
    # Heavily right skewed
  
  # acceleration
    # 6 outliers
    # seems like a normal distribution
  
  



############################################################################################################################################################################################################################



# 4. Create a correlation matrix for all of the numeric variables.

# Doubt: corrplot doesnt give correlation for missing values hence removing rows with na's

  library(ggplot2)
  library(corrplot)
  autoMPG.datumCleanNumeric <- autoMPG.datumClean[,autoMPG.numericCols]    # Fetching the data frame with only numeric columns
  autoMPG.cor_matrix <- cor(na.omit(autoMPG.datumCleanNumeric))         # Building the correlation plot
  dev.off()                                       # Closes all the previous plot windows
  corrplot(autoMPG.cor_matrix, method="number") 




############################################################################################################################################################################################################################



# 5. Identify the columns (if any) with missing data.

  colnames(autoMPG.datumClean)[colSums(is.na(autoMPG.datumClean)) > 0]
  summary(autoMPG.datumClean$horsepower)
  # horsepower has 6 missing values

  # Remove Replacing missing Values
  library(dplyr)
  library(plyr)
  
  cleanData <- function (data_in, NA_column, column_wrt, replaceType='avg', cleanType='replace'){
    data_NA <- subset(data_in, is.na(data_in[NA_column]))
    data_NotNA <- setdiff(data_in, data_NA)

    if (cleanType=='remove'){
        return (data_NotNA)
    } 
    else if (cleanType=='replace'){
      # Find the median of the horsepower column given the column_wrt
      unq_vals <- unique(data_NA[column_wrt])
      query <- sprintf("select cylinders, %s(%s) as nwHP from data_NotNA group by cylinders having cylinders in unq_vals", replaceType, NA_column)
      newMedianTable <- sqldf(query)
      
      for (cylinder_num in newMedianTable[column_wrt][,]){
        data_NA[data_NA[column_wrt] == cylinder_num , ][NA_column] <- newMedianTable[newMedianTable[column_wrt] == cylinder_num ,]["nwHP"]
      }
      
      return (rbind.fill(data_NotNA,data_NA))
    }
    else{
      return (NULL)
    }
  }


  autoMPG.datumClean <- cleanData(autoMPG.datumCleanAlias, 
                                  'horsepower', 
                                  'cylinders', 
                                  replaceType='avg', 
                                  cleanType='remove')
  dim(autoMPG.datumClean)
  
  stopifnot(dim(autoMPG.datumClean) == dim(autoMPG.datumCleanAlias))
  stopifnot(sum(is.na(autoMPG.datumClean)) == 0)

  
  
############################################################################################################################################################################################################################



# 6. Divide the data into a train/test set (80% and 20% respectively) using stratified sampling
# Performing Stratified sampling on the MPG column.
  
  library('caret')
  trainPrcnt <- 0.8
  testPrnct <- 0.2
  set.seed(32455)
  trainIndices <- createDataPartition(y = autoMPG.datumClean$mpg, p = trainPrcnt, list = FALSE)
  autoMPG.trainData <- autoMPG.datumClean[trainIndices,]
  autoMPG.testData <- autoMPG.datumClean[-trainIndices,]
  stopifnot(nrow(autoMPG.trainData) + nrow(autoMPG.testData) == nrow(autoMPG.datumClean))
  head(autoMPG.trainData)
  dim(autoMPG.trainData)
  head(autoMPG.testData)
  dim(autoMPG.testData)
  



############################################################################################################################################################################################################################




# 7. Fit a linear model to the data using the numeric variables only. Calculate the R**2 on the test set.

# Caution : ----------------------------------------------
# 1: The NA rows pertaining to the horsepower columns are removed or replaced by the median value corresponding to the num of cylinder of the vehicles.
# Caution 2: The formula implemented for SSE, SSR, SST is correct, however SST != SSE + SSR (Have to check the whats causing the mismatch)
# -------------------------------------------------------- 
  
  
# Doubt : ------------------------------------------------
# 1: "mpg" and displacement has a strong negative correlation then why is it not captured significant in the model selection. Is it because other numerical columns such as weight and horsepoewer have more affecets to "mpg" therefore the affect caused by the displacement is not significant? Could it also rise because of multicolinearity?
# -------------------------------------------------------- 
  
  
# Findings : ---------------------------------------------
# 1 -> The column acceleration is well spread and doesn't affect the column "mpg" much as other column like weight and horsepower do. Hence it is reasonable to assume that it wouldn't affect mpg as much as the other columns.

# 2 -> displacement, horsepower and weight have a slight curvature relashionship with mpg, 

# 3 -> mpg spreads out as acceleration increases  (best guess the residues might have damp out affects)

# 4 -> As it can be seen that the p-value for acceleration and displacement is too high, which states these features are not significant enough. Horsepower is significant at 90%
  
# 5 -> When Removing the NA rows:
  # By removing rows Horsepower the 
    # p-val of the column horsepower = 0.011, 
    # training R_sq = 0.7141, (which is not a very high number, this means that not much variance is explained by the model)
    # test r_sq1=0.6713278 (Not good performace)
    # test r_sq1=0.6758831
  
# 6 -> When Replacing the NA rows with the median horsepower given the number of cylinders.
  # By replacing rows with median Horsepower the 
    # p-val of the column horsepower = 0.0784, 
    # training R_sq = 0.7186, 
    # test r_sq=0.5932253 (Even worse perfomance than when removing the NA's)
# --------------------------------------------------------   

  autoMPG.trainDataNumeric <- autoMPG.trainData[, autoMPG.numericCols]
  autoMPG.testDataNumeric <- autoMPG.testData[, autoMPG.numericCols]
  summary(autoMPG.trainDataNumeric)
  summary(autoMPG.testDataNumeric)

  
  # Plotting
  dev.off() 
  par(mfrow=c(2,2))
  plot(mpg~displacement, autoMPG.trainDataNumeric)
  plot(mpg~horsepower, autoMPG.trainDataNumeric)
  plot(mpg~weight, autoMPG.trainDataNumeric)
  plot(mpg~acceleration, autoMPG.trainDataNumeric)
  
  
  
  autoMPG.numeric.model.lin <- lm(mpg~displacement + horsepower + weight + acceleration, data=autoMPG.trainDataNumeric)
  summary(autoMPG.numeric.model.lin)

  
  # Prediction on test data
  autoMPG.testDataNumeric$mpg_hat <- predict(autoMPG.numeric.model.lin, autoMPG.testDataNumeric)
  
  # Calculating R_squared for test data.
  autoMPG.testDataNumeric$residue <- autoMPG.testDataNumeric$mpg_hat - autoMPG.testDataNumeric$mpg
  autoMPG.numeric.model.lin.SSE <- sum((autoMPG.testDataNumeric$residue)^2)
  autoMPG.numeric.model.lin.SSR <- sum((autoMPG.testDataNumeric$mpg_hat - mean(autoMPG.testDataNumeric$mpg))^2)
  autoMPG.numeric.model.lin.SST <- sum((autoMPG.testDataNumeric$mpg - mean(autoMPG.testDataNumeric$mpg))^2)
  
  
  # Check if the SSE, SSR and SST are correct using SST = SSR + SSR
  stopifnot(autoMPG.numeric.model.lin.SSE + autoMPG.numeric.model.lin.SSR == autoMPG.numeric.model.lin.SST)
  
  # Calculate R_squared
  autoMPG.numeric.model.r_sq1 <- autoMPG.numeric.model.lin.SSR / autoMPG.numeric.model.lin.SST
  autoMPG.numeric.model.r_sq1
  autoMPG.numeric.model.r_sq2 <- autoMPG.numeric.model.lin.SSR / (autoMPG.numeric.model.lin.SSE+autoMPG.numeric.model.lin.SSR)
  autoMPG.numeric.model.r_sq2

  


############################################################################################################################################################################################################################




# 8. Programmatically identify and remove the non-significant variables (alpha = .05). Fit a new model with those variables removed.
# Calculate the R**2 on the test set with the new model. Did this improve performance?

# Caution : ---------------------------------------------
# 1: The NA rows pertaining to the horsepower columns are removed or replaced by the median value corresponding to the num of cylinder of the vehicles.
# Caution 2: The formula implemented for SSE, SSR, SST is correct, however SST != SSE + SSR (Have to check on whats causing the mismatch)
# -------------------------------------------------------- 
  
  
# Findings : ---------------------------------------------
# 1 -> When Removing the NA rows:
  # By removing rows Horsepower the 
    # p-val of the column horsepower = 0.000412,  (Very significant)
    # training R_sq = 0.7134, (which is not a very high number, this means that not much variance is explained by the model)
    # test r_sq=0.6758176 (The r_sq compared to the complete model improvs but negligibly)
    # test r_sq=0.6780468
  
    # The R_sq did not increase mainly because when we our the significant model contains "Horsepower" and "weight" and we know horsepwer and weight are highly correlated to the response variable "mpg" so while we removed non-significant columns such as "displacement" and "acceleration", we are not providing anything new to the model to learn. So we cant expect tht model to perform better.
  
# 2 -> When Replacing the NA rows with the median horsepower given the number of cylinders.
  # By replacing rows with median Horsepower the 
    # p-val of the column horsepower = 0.0784, 
    # training R_sq = 0.7186, 
    # test r_sq=0.5932253 (Even worse perfomance than when removing the NA's)
# --------------------------------------------------------   
  
  
  getSignificantFeatures <- function(inputSummary, dependent, alpha){
    signFeatureIndice <- which(inputSummary['Pr(>|t|)']<=alpha)
    signFeatureIndice <- signFeatureIndice[signFeatureIndice!=1]   # remove the intercept column
    return (c(row.names(inputSummary[signFeatureIndice,]), dependent))
  }
  
  # Get the signifiant Features
  significantFeatures <- getSignificantFeatures(as.data.frame(summary(autoMPG.numeric.model.lin)$coefficients), dependent="mpg", alpha=0.05)
  
  autoMPG.trainDataNumericSignificant <- autoMPG.trainDataNumeric[significantFeatures]
  autoMPG.testDataNumericSignificant <- autoMPG.testDataNumeric[significantFeatures]
  
  
  # Fit a linear model on the significant subset
  autoMPG.numeric.model2.lin <- lm(mpg~horsepower + weight,  data=autoMPG.trainDataNumericSignificant)
  summary(autoMPG.numeric.model2.lin)
  
  # Fit the model to a test Dataset
  autoMPG.testDataNumericSignificant$mpg_hat <- predict(autoMPG.numeric.model2.lin, autoMPG.testDataNumericSignificant)
  
  # Calculating R_squared for test data.
  autoMPG.testDataNumericSignificant$residue <- autoMPG.testDataNumericSignificant$mpg_hat - autoMPG.testDataNumericSignificant$mpg
  autoMPG.numeric.model2.lin.SSE <- sum((autoMPG.testDataNumericSignificant$residue)^2)
  autoMPG.numeric.model2.lin.SSR <- sum((autoMPG.testDataNumericSignificant$mpg_hat - mean(autoMPG.testDataNumericSignificant$mpg))^2)
  autoMPG.numeric.model2.lin.SST <- sum((autoMPG.testDataNumericSignificant$mpg - mean(autoMPG.testDataNumericSignificant$mpg))^2)
  
  
  # Stop If SST ! = SSE + SSR
  stopifnot(autoMPG.numeric.model2.lin.SSE + autoMPG.numeric.model2.lin.SSR == autoMPG.numeric.model2.lin.SST)
  
  
  # Calculate the r_sq
  autoMPG.numeric.model2.r_sq1 <- autoMPG.numeric.model2.lin.SSR / autoMPG.numeric.model2.lin.SST
  autoMPG.numeric.model2.r_sq1
  autoMPG.numeric.model2.r_sq2 <- autoMPG.numeric.model2.lin.SSR / (autoMPG.numeric.model2.lin.SSE + autoMPG.numeric.model2.lin.SSR)
  autoMPG.numeric.model2.r_sq2

  
  
  

############################################################################################################################################################################################################################


  
  

# 9. Attempt to fit a model on all of the relevant independent variables (including carName).
# Then calculate the R**2 on a test set. You will likely encounter an error.
# Explain why this error occurs. Fix this error.

# Caution : ----------------------------------------------
# 1: The NA rows pertaining to the horsepower columns are removed from the dataset.
# 2: The formula implemented for SSE, SSR, SST is correct, however SST != SSE + SSR (Have to check the whats causing the mismatch)
# -------------------------------------------------------- 
  
# Reason for the error:  ---------------------------------
  # The error shows up because the feature car_name is equivallent to but not exactly a primary key that is unique for every row. Hence the car name in the train data set is completely different from the car_name is the test dataset. Therefore while fitting the trained model to the test sample, the regression model is unable to recorgnize the car_name is the test data. Therefore it throws an error
  
  # To fix this problem we have to remove the column car_name from our train ans test sample data set refit the model
# --------------------------------------------------------
  
  
  sapply(autoMPG.datumVarDtype, class)
  autoMPG.datumVarDtypeClean <- na.omit(autoMPG.datumVarDtype)
  summary(autoMPG.datumVarDtypeClean)
  autoMPG.datumVarDtypeClean$car_name
  
  autoMPG.trainDataBad <- autoMPG.datumVarDtypeClean[trainIndices,]
  autoMPG.testDataBad <- autoMPG.datumVarDtypeClean[-trainIndices,]
  stopifnot(nrow(autoMPG.trainDataBad) + nrow(autoMPG.testDataBad) == nrow(autoMPG.datumVarDtypeClean))
  
  
  yVariable <- "mpg"
  xVariables <- c(names(autoMPG.trainDataBad)) 
  xVariables <- xVariables[xVariables!= yVariable]
  
  # Fit the Model on Training Sample
  autoMPG.bad.model <- as.formula(paste(yVariable, "~", paste(xVariables, collapse = '+ ')))
  autoMPG.bad.model.lin <- lm(autoMPG.bad.model, data=autoMPG.trainDataBad)
  summary(autoMPG.bad.model.lin)
  
  # Use the trained model on Test Sample    [Shows Error at this Point]
  autoMPG.testDataBad$mpg_hat <- predict(autoMPG.bad.model.lin, autoMPG.testDataBad)
  

  
  # Fixing1 : ---------------------------------------------
    xVariables <- xVariables[xVariables!= "car_name"]
    autoMPG.fixed.model <- as.formula(paste(yVariable, "~", paste(xVariables, collapse = '+ ')))
    autoMPG.fixed.model.lin <- lm(autoMPG.fixed.model, data=autoMPG.trainDataBad)
    summary(autoMPG.fixed.model.lin)
    
    
    # Use the trained model on Test Sample    [Shows Error at this Point]
    autoMPG.testDataFixed <- autoMPG.testDataBad
    autoMPG.testDataFixed$car_name <- NULL                    # We dont need to do this (this is just redundant use of memory), this is just done for the continuity of the variables
    autoMPG.testDataFixed$mpg_hat <- predict(autoMPG.fixed.model.lin, autoMPG.testDataFixed)
    
    # Calculating R_squared for the fixed Test Data
    autoMPG.testDataFixed$residue <- autoMPG.testDataFixed$mpg_hat - autoMPG.testDataFixed$mpg
    autoMPG.fixed.model.lin.SSE <- sum((autoMPG.testDataFixed$residue)^2)
    autoMPG.fixed.model.lin.SSR <- sum((autoMPG.testDataFixed$mpg_hat - mean(autoMPG.testDataFixed$mpg))^2)
    autoMPG.fixed.model.lin.SST <- sum((autoMPG.testDataFixed$mpg - mean(autoMPG.testDataFixed$mpg))^2)
    
    
    # Check if the SSE, SSR and SST are correct using SST = SSR + SSR
    stopifnot(autoMPG.fixed.model.lin.SSE + autoMPG.fixed.model.lin.SSR == autoMPG.fixed.model.lin.SST)
    
    # Calculate R_squared
    autoMPG.fixed.model.r_sq <- autoMPG.fixed.model.lin.SSR / autoMPG.fixed.model.lin.SST
    autoMPG.fixed.model.r_sq
    # r_sq = 0.6713278  .  Not a good model as the r_sq further decreases in test data, a cause for overfitting
    autoMPG.fixed.model.r_sq <- autoMPG.fixed.model.lin.SSR/(autoMPG.fixed.model.lin.SSR + autoMPG.fixed.model.lin.SSE)

    
# Fixing2 : ---------------------------------------------
    library(RecordLinkage)
    car_names <- autoMPG.datumVarDtype$car_name
    
    # brandDataframe 
    brandName <- function (carname){
      cname <- as.character(carname)
      initials <- unlist(strsplit(cname, " "))[1]
      initialList <- c(initialList, initials)
      if (initials == 'vw'){
        return("volkswagen")
      }
      return (initials)
    }
    
    initials <- sapply(car_names, FUN=brandName)
    
    # Store it into a dataframe:
    initials <- data.frame(initials)

    uniqueCount <- sqldf("select initials as brand, count(*) as occurance from initials group by initials")  # We see that some spellings are wrong
    
    sapply(uniqueCount["brand"], class)
    uniqueBrands <- c(uniqueCount["brand"])
    uniqueBrands$brand
  
    library('stringdist')
    library('dplyr')
    # The below script just finds the carname that are mis - spelled
    jaroDistMatrix <- 1-stringdistmatrix(uniqueBrands$brand,useNames="strings",method="jw")
    kpm <- data.frame(as.matrix(jaroDistMatrix))
    
    simThresh <- 0.8
    idx <- apply(kpm, 2, function(x) x >0.8)
    idx <- apply(idx, 1:2, function(x) if(isTRUE(x)) x<-1 else x<-NA)
    matrix_abs <- na.omit(melt(idx)) 
    
    # Now we replace all the 
    
    sqldf("select a.Var1, b.Var1 from matrix_abs as a join matrix_abs as b on a.Var1=b.Var2 and a.var2=b.var1")
    
    
    
############################################################################################################################################################################################################################


  
  

# 10. Determine the relationship between model year and mpg.
# Interpret this relationship.
# Theorize why this relationship might occur.
  
    
# Theory : -----------------------------------------------
    # As it can be seen from the regression output that, the model_year 74,75,,76,77,78,79,80,81,82 are significant with a 96% confidence, which also says that "model_year" explains the variation in "MPG" and hence is related to the "MPG". As a complement to the regression model it can also be viewed from the groupTable and the scatter plot that despite the "mpg" for different years are speread across many values, the data follows a pattern i.e. Every year the min value of "mpg" and maximum value of "mpg" marginally increases from its preceding year (with an exception of 1971, 1979 and 1981). This relationship makes sense because significant improvement in technology every year would foster more feul efficient vehicles.
    
    # The chi-square value produces a p-value very less, 
    # H0: The categorical column ntile(mpg) and the model_year are independent of each other
    # Ha: The categorical column ntile(mpg) and the model_year are not dependent of each other
    # By viewing the p-value, we can safely reject the null hypothesis that the ntile(mpg) depends on the column model_year
#  --------------------------------------------------------
    
    
  # Test 1 : Using Regression 
  regression.Model <- lm(autoMPG.datumClean$mpg~autoMPG.datumClean$model_year)
  summary(regression.Model)

  
  # Test 2 : Using scatterplot 
  # Use group by command to find min, max and range of all "mpg's" pertaining to each "model_year"
  library(sqldf)
  aliasDatumVarDtype <- autoMPG.datumVarDtype
  groupTable <- sqldf("select model_year, count(mpg) as count, min(mpg) as min, max(mpg) as max, max(mpg)-min(mpg) as range from aliasDatumVarDtype group by model_year")
  groupTable
  
  # First Convert
  modelYear <- as.integer(as.character(autoMPG.datumVarDtype$model_year))
  dev.off()
  plot(modelYear, autoMPG.datumVarDtype$mpg)
  head(as.data.frame(c(modelYear,autoMPG.datumVarDtype$mpg)))
  

  # Test 3 : Using Chi-Square Test
  library(dplyr)
  autoMPG.datumClean$mpgQuartile <- ntile(autoMPG.datumClean$mpg, 4)
  tbl <- table(autoMPG.datumClean$mpgQuartile, autoMPG.datumClean$model_year)
  chisq.test(tbl)
  # summary: X-squared = 196.88, df = 36, p-value < 2.2e-16
  




  
  

############################################################################################################################################################################################################################


  
  
# 11. Build the best linear model you can (as measured by R**2 on the test data)
# Record the value obtained in the comments below. Make sure to show all your code.

# Theory : ---------------------------------------------
  # 1: A good model is a model thats small and doesn't overfit (Ocams Razor). i.e the difference between the training and test R_sq is not high, rather the R_sq for test should be high.
  
  # 2: From the excerise 10 we see that the feature model_year has some relationship with the dependent variable mpg. Hence we may wanna consider having model_year in our model.
  
  # 3: From all the correlation plots we have already seen that the "mpg" column is very highly correlated (-vely or positively) to the columns "horsepower", "displacement", "cylinder", "weight". Hence it can be intuitively thought as that taking all these variables into our model may give rise to multicolinearity that may perform bad for the test data.
  
  # 4: Moreover from fitting models in the earlier section we have seen that acceleration was always shown to be non-significant variable and doesnt contribute to the model much.
  
  # 5: So we fix the other independent variables ("model_year" and "origin") and then step-by-step evaluate other features by adding them to the model.
  
  # 6: Since we see that many of the features set are skewed so doing a log transformation would help. Lets do it now.
#-------------------------------------------------------

  
  buildModel <- function(model, trainDataIN, testDataIN){
      linFit <- lm(model, data=trainDataIN)
      y_hat <- predict(linFit, testDataIN)
      residue <- y_hat - testDataIN$mpg
      SSE <- sum((residue)^2)
      SSR <- sum((y_hat - mean(testDataIN$mpg))^2)
      SST <- sum((testDataIN$mpg - mean(testDataIN$mpg))^2)
      SST2 <- SSE + SSR
      
      return (list(linFit, y_hat, residue, SSE, SSR, SST, SST2))
  }
  
  
  # Create DataSet:
      # First lets see the Log transformation of the numeric columns in the dataCleaned Dataset
      library(reshape2)
      dev.off() 
      ggplot(data = melt(log(autoMPG.datumClean[autoMPG.numericCols])), mapping = aes(x = value)) + 
        geom_histogram(bins = 10) + facet_wrap(~variable, scales = 'free_x')
      
      # The log transformation looks much better, now let's builf the dataframe for LOG Transformation
      
      # Use Stratified sampling to split the data into train and test model:
      trainPrcnt <- 0.8
      testPrnct <- 0.2
      # set.seed(32455)
      set.seed(4673)
      
      # Noramal Test Train Data
      trainIndicesNew <- createDataPartition(y = autoMPG.datumClean$mpg, p = trainPrcnt, list = FALSE)
      autoMPG.trainDataNew <- autoMPG.datumClean[trainIndicesNew,]
      autoMPG.testDataNew <- autoMPG.datumClean[-trainIndicesNew,]
      stopifnot(nrow(autoMPG.trainDataNew) + nrow(autoMPG.testDataLog) == nrow(autoMPG.datumClean))
      head(autoMPG.trainDataNew)
      head(autoMPG.testDataNew)
      dim(autoMPG.trainDataNew)
      dim(autoMPG.testDataNew)
      
      # Log Transformed Test Train Data
      autoMPG.dataLogTransformed <- autoMPG.datumClean
      autoMPG.dataLogTransformed[,autoMPG.numericCols] <- log(autoMPG.dataLogTransformed[,autoMPG.numericCols])
      head(autoMPG.dataLogTransformed)
      trainIndicesLog <- createDataPartition(y = autoMPG.dataLogTransformed$mpg, p = trainPrcnt, list = FALSE)
      autoMPG.trainDataLog <- autoMPG.dataLogTransformed[trainIndicesLog,]
      autoMPG.testDataLog <- autoMPG.dataLogTransformed[-trainIndicesLog,]
      stopifnot(nrow(autoMPG.trainDataLog) + nrow(autoMPG.testDataLog) == nrow(autoMPG.dataLogTransformed))
      head(autoMPG.trainDataLog)
      head(autoMPG.testDataLog)
      dim(autoMPG.trainDataLog)
      dim(autoMPG.testDataLog)
      
  
  # MODEL 1:  (Normal - Untransformed Dataset) ---------------------------------------------
      # Output:
          # Model: mpg ~ model_year + origin + horsepower
          # training R_sq = 0.7917
          # testing R_sq2 = 0.84316
          # testing R_sq2 = 0.8182
    
      xVariables <- c("model_year", "origin", "horsepower")
      yVariable <- "mpg"
      autoMPG.best.model1 <- as.formula(paste(yVariable, "~", paste(xVariables, collapse = '+ ')))
      model1.output <- buildModel(autoMPG.best.model1, autoMPG.trainDataNew, autoMPG.testDataNew)
      
      summary(model1.output[[1]])
      autoMPG.testDataNew$mpg_hat <- model1.output[[2]]
      autoMPG.testDataNew$residue <- model1.output[[3]]
      head(autoMPG.testDataNew)
      autoMPG.best.model1.lin.SSE <- model1.output[[4]]
      autoMPG.best.model1.lin.SSR <- model1.output[[5]]
      autoMPG.best.model1.lin.SST <- model1.output[[6]]
      autoMPG.best.model1.lin.SST2 <- model1.output[[7]]


      # Check if the SSE, SSR and SST are correct using SST = SSR + SSR
      stopifnot(autoMPG.best.model1.lin.SSE + autoMPG.best.model1.lin.SSR == autoMPG.best.model1.lin.SST)

      # Calculate R_squared
      autoMPG.best.model1.r_sq1 <- autoMPG.best.model1.lin.SSR / autoMPG.best.model1.lin.SST
      autoMPG.best.model1.r_sq1
      autoMPG.best.model1.r_sq2 <- autoMPG.best.model1.lin.SSR / autoMPG.best.model1.lin.SST2
      autoMPG.best.model1.r_sq2


  
  
  # BEST Model 2 (Log Transformed):   ---------------------------------------------
      # Output: 
          # Model: mpg ~ cylinders + displacement + horsepower + weight + acceleration + model_year + origin
          # training_Data: R_square = 0.9188
          # test_data: R_sq1 = 0.8978753: 
          # test_data: R_sq1 = 0.9028319:
          # Thought the test R_sq is less than the tran R_sq, the model is much better compared to others.
  
        xVariables <- c("cylinders", "displacement","horsepower", "weight", "acceleration", "model_year", "origin")
        yVariable <- "mpg"
        autoMPG.best.model2 <- as.formula(paste(yVariable, "~", paste(xVariables, collapse = '+ ')))
        model2.output <- buildModel(autoMPG.best.model2, autoMPG.trainDataLog, autoMPG.testDataLog)
        
        summary(model2.output[[1]])
        autoMPG.testDataLog$mpg_hat1 <- model2.output[[2]]
        autoMPG.testDataLog$residue1 <- model2.output[[3]]
        head(autoMPG.testDataLog)
        autoMPG.best.model2.lin.SSE <- model2.output[[4]]
        autoMPG.best.model2.lin.SSR <- model2.output[[5]]
        autoMPG.best.model2.lin.SST <- model2.output[[6]]
        autoMPG.best.model2.lin.SST2 <- model2.output[[7]]
        
        
        # Check if the SSE, SSR and SST are correct using SST = SSR + SSR
        stopifnot(autoMPG.best.model2.lin.SSE + autoMPG.best.model2.lin.SSR == autoMPG.best.model2.lin.SST)
        
        # Calculate R_squared
        autoMPG.best.model2.r_sq1 <- autoMPG.best.model2.lin.SSR / autoMPG.best.model2.lin.SST
        autoMPG.best.model2.r_sq1
        autoMPG.best.model2.r_sq2 <- autoMPG.best.model2.lin.SSR / autoMPG.best.model2.lin.SST2
        autoMPG.best.model2.r_sq2
        

        
  # BEST Model 3: (Log Transformed)  ---------------------------------------------
      # Output: 
          # Model: mpg ~ model_year + cylinders + horsepower + displacement
          # training_Data: R_square = 0.9113
          # test_data: R_sq1 = 0.9007224: 
          # test_data: R_sq1 = 0.9000995:
          # Thought the test R_sq is less than the tran R_sq, the model is much better compared to others.
        
        yVariable <- "mpg"
        xVariables <- c("model_year", "cylinders","weight","displacement")
        autoMPG.best.model3 <- as.formula(paste(yVariable, "~", paste(xVariables, collapse = '+ ')))
        model3.output <- buildModel(autoMPG.best.model3, autoMPG.trainDataLog, autoMPG.testDataLog)
        
        summary(model3.output[[1]])
        autoMPG.testDataLog$mpg_hat3 <- model3.output[[2]]
        autoMPG.testDataLog$residue3 <- model3.output[[3]]
        head(autoMPG.testDataLog)
        autoMPG.best.model3.lin.SSE <- model3.output[[4]]
        autoMPG.best.model3.lin.SSR <- model3.output[[5]]
        autoMPG.best.model3.lin.SST <- model3.output[[6]]
        autoMPG.best.model3.lin.SST2 <- model3.output[[7]]
        
        
        # Check if the SSE, SSR and SST are correct using SST = SSR + SSR
        stopifnot(autoMPG.best.model3.lin.SSE + autoMPG.best.model3.lin.SSR == autoMPG.best.model3.lin.SST)
        
        # Calculate R_squared
        autoMPG.best.model3.r_sq1 <- autoMPG.best.model3.lin.SSR / autoMPG.best.model3.lin.SST
        autoMPG.best.model3.r_sq1
        autoMPG.best.model3.r_sq2 <- autoMPG.best.model3.lin.SSR / autoMPG.best.model3.lin.SST2
        autoMPG.best.model3.r_sq2
        
  
  # BEST Model 4: (Log Transformed)  ---------------------------------------------
    # Output: 
        # Model: mpg ~ model_year + cylinders + horsepower + displacement
        # training_Data: R_square = 0.9016
        # test_data: R_sq1 = 0.8806303: 
        # test_data: R_sq1 = 0.8953331:
        # Thought the test R_sq is less than the tran R_sq, the model is much better compared to others.
        
        yVariable <- "mpg"
        xVariables <- c("model_year", "weight","displacement")
        autoMPG.best.model4 <- as.formula(paste(yVariable, "~", paste(xVariables, collapse = '+ ')))
        model4.output <- buildModel(autoMPG.best.model4, autoMPG.trainDataLog, autoMPG.testDataLog)
        
        summary(model4.output[[1]])
        autoMPG.testDataLog$mpg_hat4 <- model4.output[[2]]
        autoMPG.testDataLog$residue4 <- model4.output[[3]]
        head(autoMPG.testDataLog)
        autoMPG.best.model4.lin.SSE <- model4.output[[4]]
        autoMPG.best.model4.lin.SSR <- model4.output[[5]]
        autoMPG.best.model4.lin.SST <- model4.output[[6]]
        autoMPG.best.model4.lin.SST2 <- model4.output[[7]]
        
        
        # Check if the SSE, SSR and SST are correct using SST = SSR + SSR
        stopifnot(autoMPG.best.model4.lin.SSE + autoMPG.best.model4.lin.SSR == autoMPG.best.model4.lin.SST)
        
        # Calculate R_squared
        autoMPG.best.model4.r_sq1 <- autoMPG.best.model4.lin.SSR / autoMPG.best.model4.lin.SST
        autoMPG.best.model4.r_sq1
        autoMPG.best.model4.r_sq2 <- autoMPG.best.model4.lin.SSR / autoMPG.best.model4.lin.SST2
        autoMPG.best.model4.r_sq2
        
        