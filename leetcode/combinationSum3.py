

def combinationSum3(k, n):
    arr = [0,1,2,3,4,5,6,7,8,9]
    i =0
    j = 8

    
    list1 = []
    list2 = []
    for k_val in range(0,k):
        sum = arr[i]
        sum += arr[j]
        if n - sum < 2:
            i +=1
        elif n-sum>2:
            j -= 1
        else:
            print (i, j, arr[i], arr[j])
            
        # if k_val == k-1:
        #     n - sum >=2:
        #     i
        
combinationSum3(k=4, n = 27)