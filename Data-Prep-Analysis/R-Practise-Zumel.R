addwd('/Users/sam/All-Program/App-Dataset/Study/IIT-Code/Data-Prep-Analysis')

load('/Users/sam/All-Program/App-Dataset/Study/IIT-Code/Data-Prep-Analysis/phsample.RData')
# The Data set is loaded into the R and is available with the name of dpus


# The below code will select a subset of the entire dpus data given the conditions below are matched
psub = subset(dpus, with(dpus,(PINCP>1000)
                         &(ESR==1) 
                         &(PINCP<=250000)
                         &(PERNP>1000)
                         &(PERNP<=250000)
                         &(WKHP>=40)
                         &(AGEP>=20)
                         &(AGEP<=50)
                         &(PWGTP1>0)
                         &(COW %in% (1:7))
                         &(SCHL %in% (1:24))))

# encoding as factor
head(psub$SEX)
psub$SEX = as.factor(ifelse(psub$SEX==1, 'M', 'F'))  # If psub$SEX == 1 then 'M' else 'F'
psub$SEX = relevel(psub$SEX,'M')

# Working on COW column
unique(psub$COW)
# creating a mapping for the COW column with values 1,2,3,4,5,6,7
cowmap <- c("Employee of a private for-profit",  # when 1
            "Private not-for-profit employee",   # when 2
            "Local government employee",         # when 3
            "State government employee",
            "Federal government employee",
            "Self-employed not incorporated",
            "Self-employed incorporated")
# Changing the labels of the COW columns and converting it into factor
psub$COW = as.factor(cowmap[psub$COW])    # psub$COW will produce numbers as 1,2,3,4,5,6,7
psub$COW = relevel(psub$COW,cowmap[1])


# Working on the SCHL column
schlmap = c(rep("no high school diploma",15),
  "Regular high school diploma",
  "GED or alternative credential",
  "some college credit, no degree",
  "some college credit, no degree",
  "Associate's degree",
  "Bachelor's degree",
  "Master's degree",
  "Professional degree",
  "Doctorate degree")
# Changing the labels of the SCHL column and converting into factor
psub$SCHL = as.factor(schlmap[psub$SCHL])
psub$SCHL = relevel(psub$SCHL,schlmap[1])



# Create a training and test set
dtrain = subset(psub,ORIGRANDGROUP >= 500)
dtest = subset(psub,ORIGRANDGROUP < 500)
summary(dtrain$COW)
