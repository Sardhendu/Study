funfunction <- function(a=10,b=11,c=12){
  return (list(a=a, b=b, c=c))
}


# 1. a <- 2 for assignment
# 2. a[3:5] display the value of array 3,4,5
# 3. a[c(1,4, length(a))]  captures the object in array 1, 4 and the last value


# 1. funfunction(1,2,3)
# 2. funfunction(1,4)
# 3. args(funfunction)

# x <- 1:12  # Forms an array from 1 o 12
# dim(x) <- c(3,4)  # converts the array of lenght 12 to a matrix of dimension 3,4
# dim(x) <- c(1,2)  # wont work as x has length 12 and the matrix cell numbers should be 12 
                    # only c(1,12), c(2,6), c(3,4), c(4,3), c(6,2) and c(12,1) satisfies the number of cells
                    # we can also have c(2,2,3) its combination.
# x <- matrix(1:12, nrow=3, byrow=TRUE)  # byrow=TRUE will keep numbers ascending to escending horizontaly
                                        # byrow=TRUE will keep numbers ascending to escending vertically

# Matrix operation:
# cell by cell addition, subtraction, division, multiplication = x+y, x-y, x/y, x*y
# matrix multiplication = x%*%y
# matrix transpose = t(x)

# Factors(Labels) for Categorical outputs:
# lap_review = c(5,5,4,2,1,3,4,2,1,3,5,5,3)  # provides labels for a laptop review based on stars
# f_lab_review = factor(lap_review, levels = 1:5)
# levels(f_lap_review) <- c("worst", "unsatisfactory", "okay", "good", "satisfactory") 
      # creates an array of categorical labels and assigns the labels to the data set or maps 
      # 1 to worst, 2 to unsatisfactory, 3 to okay, 4 to good, 5 to satisfactory

