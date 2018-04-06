


def numSubarrayProductLessThanK1(nums, k):
    # NOT VERY EFFICIENT
    count = 0
    while len(nums) > 0:
        first = nums[0]
        # print ('first_num ', first)
        nums.pop(0)
        if first <k:
            count += 1
        keep_prod = first
        for val in nums:
            # print ('val ', val)
            if val * keep_prod < k:
                keep_prod = val*keep_prod
                count += 1
            else:
                break
            # print('current count :', count)
    # print (count)
        # num.pop(0)
    return count
    
    
def numSubarrayProductLessThanK(nums, k):
    i=0
    l = 0
    m = 0
    count = 0
    ext_arr = nums
    keep_prod = nums[0]
    nw_arr = []
    while i<len(nums):
        
        element = nums[i]
        nw_arr.append(element)
        
        print('cap number ', element)
        for j in range(i+1, len(nums)):
            print ('input j is : ', j)
            print('keep_prod is:', keep_prod)
            val = nums[j]
            print('s val ', val)
            if val * keep_prod < k:
                nw_arr.append(val)
                keep_prod = val*keep_prod
                print('yeyesyesyes: ', keep_prod)
            else:
                count += len(nw_arr)
                keep_prod = keep_prod/ext_arr[0]
                ext_arr.pop(0)
                print ('yoworiewoiruweioruioweuior ', j, nw_arr)
                nw_arr.pop(0)
                break
        i = j - 1
        
        if l == 10:
            break

        l += 1

    
print (numSubarrayProductLessThanK([10, 5, 2, 3, 1, 1, 100, 10], 310))
