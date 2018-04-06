


def maxProduct(nums):
    prod = 1
    from_idx = 0
    count = 0
    i = 0
    from_to = [0,1]
    nums.append(-1)
    # print(nums)
    while i<len(nums):
        a = prod * nums[i]
        # print ('popopopopopo ', a)
        if a< prod:
            # print ('afhsdgfdshhdgfdgfrom_to')
            prod = 1
            if i - from_idx > count:
                from_to = [from_idx, i]
            count = i - from_idx
            from_idx = i + 1
        else:
            prod = a
        
        # print (from_idx, i, count, prod)
        
        # if count
        i += 1
        
    return nums[from_to[0]:from_to[1]][0]
        
    
        
maxProduct(nums=[-2])#[0,2,3,-3,6,1,2,-5,2,1,3,17,0])