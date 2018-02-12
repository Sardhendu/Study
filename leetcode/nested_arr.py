



def arrayNesting(nums):
    visited = set()
    i = 0
    idx = 0
    val = nums[idx]
    keep_max = 0
    max_idx = 0
    while i<=len(nums)-1:
        # print ('Runnning i is:', i)
        idx = i
        count = 0
        while idx not in visited:
            # print ('Running idx :', idx)
            visited.add(idx)
            idx = nums[idx]
            count+=1
        keep_max = max(keep_max, count)
        max_idx=idx if count > keep_max else max_idx
        i+=1

    
    nw_arr = []
    val = max_idx
    nw_arr.append(nums[max_idx])
    max_idx = nums[max_idx]
    # print (max_idx)
    while max_idx != val:
        nw_arr.append(nums[max_idx])
        max_idx = nums[max_idx]
    
    return keep_max, nw_arr

arrayNesting(nums=[5,4,0,3,1,6,2])