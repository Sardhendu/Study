
def productExceptSelf(nums):
    p = 1
    n = len(nums)
    output = []
    for i in range(0 ,n):
        output.append(p)
        p = p * nums[i]
    print (output)
    print (p)
    p = 1
    for i in range(n -1 ,-1 ,-1):
        print ('p: ', p)
        print ('nums[i]: ', nums[i])
        print ('opt: ', output[i]*p)
        output[i] = output[i] * p
        p = p * nums[i]
        
    print (output)
    return output


productExceptSelf([1,2,3,4])