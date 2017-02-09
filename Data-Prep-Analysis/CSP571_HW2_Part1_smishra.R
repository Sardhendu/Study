rm(list = ls())

# Please note that Homework Two consists of two parts, an R file and a
# Text document. Ensure that you submit both parts.
# Load in the Boston Housing data set using the code below.
# install.packages('mlbench')

library(mlbench)
data(BostonHousing)

# 1. Create a scatterplot matrix of all variables in the data set. Save your output.
head(BostonHousing)
plot(BostonHousing)



# 2. For each numeric variable in BostonHousing, create a separate boxplot using
# "Method 2" listed in the class notes. Do this programmatically; meaning do
# not simply hardcode the creation of every boxplot. Instead, loop over the
# approriate columns and create the boxplots. Save your output. Ensure your boxplots
# all have proper titles
cols <- sapply(BostonHousing, is.numeric)
numeric_cols <- names(which(cols==TRUE))        # Fetch all the numeric colums

# Create a subplot frame for plotting all the box plots using a loop,
# The method two suggest a horizontal box plot
plot.new()
par(mfrow=c(5,3))
crearteBoxPlots <- function (column_name){
  boxplot(BostonHousing[column_name], horizontal = TRUE,  main= column_name)
} 
sapply(numeric_cols, FUN=crearteBoxPlots)




# 3. Create a correlation matrix and correlation plot
# for the BostonHousing data set. Save your output.

# Doubt: Correlation matrix is to be for the entire dataset or the dataset with numeric columns only.
# Can we plot the correlation matrix using the corrplot package 
library(corrplot)
BostonHousingNumeric <- BostonHousing[,cols]    # Fetching the data frame with only numeric columns
cor_matrix <- cor(BostonHousingNumeric)         # Building the correlation plot
dev.off()                                       # Closes all the previous plot windows
corrplot(cor_matrix, method="number")           # plotting the correlation matrix





# 4. Identify the top 3 strongest absolute correlations in the data set. Save your output.
# Correlation matrix is symmetric hence, Fetching the upper triangular region of the correlation matrix and also excluding the diagonal elements.
library(reshape)

fetchTopCorrelation <- function(matrix, num_top_elements){
  matrix_abs <- abs(matrix)
  matrix_abs[lower.tri(matrix_abs, diag = TRUE)] <- NA  # Initialize all the diagonal elements and the lower triangular region to NA
  matrix_abs <- na.omit(melt(matrix_abs))   # Convert the table (With NA on lower triangular region and diagonal elements) into data frame like structure with their corresponding values and simultaneouly omit all NA's. In short this outputs a dataframw like stucture with the corelation between elements of only the upper traingular region.
  matrix_abs <- matrix_abs[order(-(matrix_abs$value)),]   # Sort the dataframe in descending order based on the column "value"- that hold the absolute correlation coefficient 
  return (matrix_abs[1:num_top_elements,] )  # Fetch the top desired elements
}

topELements = fetchTopCorrelation(cor_matrix, 3)   # Finally select the top three elements
topELements




# 5. Create a new variable call ageGroup quartiles. Divide the age variable
# into four even sections and assign it to one quartile.
library(dplyr)
BostonHousingNumeric$ageGroupquartiles <- ntile(BostonHousingNumeric$age, 4) 
# This invokes a package fucntion that automatically convert the column into 4 equal parts
head(BostonHousingNumeric)




# 6. Go to the website listed below. Convert the html table into a
# dataframe with columns NO, Player, Highlights
library('rvest')
library('tidyr')
library(gdata)
url = 'http://www.espn.com/nfl/superbowl/history/mvps'

page <- read_html(url)    # access the Url and fetch the head and the body of the page and put it into a list
superBowl_tab <- html_nodes(page, css = 'table')    # The css='table' captures the tabular part from the html page
t <- superBowl_tab[[1]]   # Get the element from the list
superBowl_dat <- html_table(superBowl_tab)[[1]]     # Finally get the data
superBowl_dat <- data.frame(superBowl_dat)[-1:-2,]  # Remove the first two rows, The second row is the column name but we hardcode it as below
names(superBowl_dat) <- c("NO", "Player", "Highlights")
head(superBowl_dat)




# 7.Extract the names of the MVPs, Position and Team into columns
# MVP1, MVP2, Position, Team
# Doubt: Would [MVP1,MVP2,position,Team] be [Bart,Starr,QB,Green Bay] for the first record
extractNmsPosTm <- function(val){
  playerName <- unlist(strsplit(val[1], "&"))
  return (c(trimws(playerName[1]), trimws(playerName[2]), trimws(val[2]), trimws(val[3])))# We dont have to worry if the string val doesnt have enough value. In such a case R automatically assigns NA as a value
}

playerInfo = strsplit(superBowl_dat$Player, ",")
mat <- sapply(playerInfo, FUN = extractNmsPosTm) 
superBowl_dat$MVP1 <- mat[1,]
superBowl_dat$MVP2 <- mat[2,]
superBowl_dat$Position <- mat[3,]
superBowl_dat$Team <- mat[4,]
head(superBowl_dat)



# 8. Determine the 90th%, 92.5th%, 95th%, 97.5th% and 99th% confidence intervals
# for the mean of passing yards
# (as listed in "Highlights" column) for quarterbacks.
# Note that there are a few intermediate steps you'll probably want to do
# in order to accomplish this. I am purposelly leaving that up to you, so that
# you are starting to develop the types of problem solving skills you'll need
# to tackle these problems in the wild.

# Doubts: 
# 1. The Data here is pretty small. So its quite visible that the pattern is "value yards passing". However in real world problems we can have errors such as spellings mistakes, the yard mentioned in different order (passing yards value)


# Function to extract the passing yard information
passsingYards <- function(val){
  if (grepl("\\d{2,3}\\s+(yards|yard)\\s+pass", val)==TRUE){
    return (as.numeric(gsub("(\\d{2,3})\\s+(yard|yards)\\s+pass.*$",'\\1', val)))
  }
# Will match any string with yard, pass and with number with 2 digits or 3 digits and extraxt the digit information
}

# Function to find the confidence Interval
CI <- function(input_data, confidence){
  n <- length(input_data)
  mean_yard <- mean(yard_arr)
  sd_yard <- sd(input_data)
  std_err <- sd_yard/sqrt(n)
  alpha <- (1-confidence)/2
  
  # Using Computing Confidence interval from the T-distribution
  int_err <- qt(confidence+alpha,df=n-1)*std_err
  
  # t.test(input_data,mu=3)
  min_thresh <- mean_yard-int_err
  max_thresh <- mean_yard+int_err
  
  print (n)
  print (alpha)
  print (confidence)
  print (mean_yard)
  print (sd_yard)
  print (std_err)
  print (min_thresh)
  print (max_thresh)
  return (list(confidence*100, min_thresh, max_thresh))
}

# Step -1 Get the data for only Quarterbacks using the Position column.
qrtrBacks <- subset(superBowl_dat,superBowl_dat$Position =='QB')

# Step -2 Use the above function to get the data for passing yards using for only quarterbacks
yard_arr <- unlist(sapply(qrtrBacks$Highlights, FUN=passsingYards), use.names = FALSE)

# Step -3 Check if their are outliers (If the plot is skewed),
boxplot(yard_arr, horizontal = TRUE, main='Length in Yard')
stripchart(yard_arr, add = TRUE, pch = 20, col = 'blue')    # The data seems good

# Step -4 Find the CI interval for the mentioned Confidence value
ConfidenceInterval <- as.data.frame(CI(yard_arr, c(0.9, 0.925, 0.95, 0.975, 0.99)), col.names = c("Confidence %", "min_thrsh","max_thresh"))
ConfidenceInterval





# 9. The following contains data on the calorie counts of four types
# of foods. Perform an ANOVA and determine the Pr(>F)
food1 <- c(164,   172,  168,  177, 	156, 	195)
food2 <- c(178,   191, 	197, 	182, 	185, 	177)
food3 <- c(175, 	193, 	178, 	171, 	163, 	176)
food4 <- c(155, 	166, 	149, 	164, 	170, 	168)

food_var <- c(rep('food1',length(food1)),    # rep will just repeat the variable name for lenght(food) times
              rep('food2',length(food2)),
              rep('food3',length(food3)),
              rep('food4',length(food4)))

values <- c(food1, food2, food3, food4)
df = data.frame(food_var, values)        # create a data frame with two columns (category name and value) because we would like to fit anova to the data
sapply(food_var, class)
sapply(values, class)
head(df)
# Fit ANOVA just like normal model fit to a data frame.
sapply(df, class)
fit <- aov(values~food_var, data=df)
fit
summary(fit)





# 10. Install the lubridate package and familarize yourseslf with it.
# This is the preferred package for handling
# dates in R, which is an important skill.
# Practing handling dates by using the package to determine how many
# Tuesdays fell on the first of the month
# during the 19th century (1 Jan 1801 to 31 Dec 1901).
install.packages('lubridate')
library('lubridate')

# Create a vectore with all the dates between the given dates
# dateVector <- seq(as.Date("2017-01-01"), as.Date("2017-03-01"), by="days")
# countJanTuesday <- sum(sapply(dateVector, FUN=getNumTuesdaysFirstMonth))
dateVector <- seq(as.Date("1801-01-01"), as.Date("1901-12-31"), by="days")
dfIfTuesday = data.frame(date=dateVector)

dfIfTuesday$Tuesday = wday(dfIfTuesday$date,label=TRUE) == "Tues"
dfIfTuesday$day1 = day(dfIfTuesday$date) == 1
numTuesday1 <- sum(dfIfTuesday$Tuesday & dfIfTuesday$day1)








##################################  PART 2 Miscellaneous (Analysis)  ##################################
# Question 11: In the Boston Housing data set, what is the relationship between crime and housing prices? Please support your claims with exploratory analysis conducted in R. Does this relationship make sense? Justify your answer. IE: What are some reasons this relationship makes sense or does not make sense?
plot(BostonHousing$crim, BostonHousing$medv)
plot(BostonHousing$crim, BostonHousing$rm)
plot(BostonHousing$rm, BostonHousing$medv)
cor(BostonHousing$medv, BostonHousing$crim)
cor(BostonHousing$medv, BostonHousing$rm)

index_lesscrime <- which(BostonHousing$crim<5)   # For less crime prone area Housing price is much better explained by the feature "rm"-> num of room per dwelling
medv_less_crime <- BostonHousing$medv[index_lesscrime]
rm_less_crime <- BostonHousing$rm[index_lesscrime]
plot(rm_less_crime, medv_less_crime)

# Fiting a linear model to observe the Coeficient of deterination to measure the significance of the features "crime" and "rm" in explaining the variance in "medv"->housing price
medv_VS_rm.lm <- lm(BostonHousing$medv ~ BostonHousing$rm)
medv_VS_crim.lm <- lm(BostonHousing$medv ~ BostonHousing$crim)
summary(medv_VS_rm.lm)
summary(medv_VS_crim.lm)


# Question 12: Based on your analysis of the Boston Housing data set, please provide an interpretation for the top 3 strongest absolute correlations. Offer some hypothesis as to why these correlations may be present. 
plot(BostonHousing$rad, BostonHousing$tax)
cor(BostonHousing$rad, BostonHousing$tax)

plot(BostonHousing$nox, BostonHousing$dis)
cor(BostonHousing$nox, BostonHousing$dis)

plot(BostonHousing$nox, BostonHousing$indus)
cor(BostonHousing$nox, BostonHousing$indus)
