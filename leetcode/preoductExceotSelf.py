

def productExceptSelf(nums):
    val = 1
    val2 = 1
    h_0 = 0
    idx = 0
    for num, i in enumerate(nums):
        if i == 0:
            h_0 += 1
            idx = num
        else:
            val2 = val2*i
        val = val*i
    print (val, val2, )
    
    if h_0 > 1:
        return [0]*len(nums)
    elif h_0 == 1:
        nums = [0] * len(nums)
        nums[idx] = val2
        return nums
    else:
        j  = 0
        while j<len(nums):
    
            nums[j] = val/nums[j]
            j+=1
        # print (nums)
        return nums

print (productExceptSelf(nums=[0,0,3,4]))