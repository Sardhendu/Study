


def maxAvgSubArray(arr,k):
    sum_first_k = sum(arr[0:k])/k
    # print (sum_first_k)
    new_sum = sum_first_k
    max_sum = sum_first_k
    
    if len(arr) > k:
        for i in range(1, len(arr)-k+1):
            j = i+k - 1
            new_sum = new_sum - (arr[i-1]/k)  + (arr[j]/k)
            max_sum = max(max_sum, new_sum)

    return max_sum


# print (maxAvgSubArray([5],1))#1,2,3,4,5], 4))