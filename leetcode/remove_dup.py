# Python3 code to find
# duplicates in O(n) time

# Function to print duplicates
def findDuplicates(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    out = []
    for i in range(0, len(nums)):
        
        # print (abs(nums[i]-1))
        # print (nums[abs(nums[i]-1)])
        # print(-1*nums[abs(nums[i]-1)])
        
        if nums[abs(nums[i]) - 1] < 0:
            # print ('Gotcha ', abs(nums[i]))
            out.append(abs(nums[i]))
        else:
            nums[abs(nums[i]) - 1] = -1 * nums[abs(nums[i]) - 1]
            # print (nums)
    return out

def findDisappearedNumbers(nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    out = []
    
    
    for i in range(0, len(nums)):
        print (abs(nums[i])-1)
        print (nums[abs(nums[i])-1])
        if nums[abs(nums[i]) - 1] < 0:
            print ('gotcha, the index is: ', i)
            pass
        else:
            nums[abs(nums[i]) - 1] = -1 * nums[abs(nums[i]) - 1]
        print (nums)
    print ('asas' ,nums)

    for i, val in enumerate(nums):
        if val >= 0:
            out.append(i + 1)
    return out
    
def missingIntegers(nums):
    tot = len(nums) + 1
    out = []
    for i in range(0, len(nums)):
        if abs(nums[i])-tot < 0:
            nums[abs(nums[i]) - 1] = abs(nums[abs(nums[i]) - 1]) + tot
        else:
            act = nums[i] - tot
            print (act)
            nums[act - 1] = nums[act - 1] + tot

    for i in range(0, len(nums)):
        if nums[i] <= tot:
            out.append(i+1)
    return out


def firstMIssingPositiveInteger(nums):
    max = len(nums)
    min = 0
    
    for i in range(0,len(nums)):
        if (nums[i] - 1 < min) or (nums[i] - 1 >= max):
            nums[i] = 0
        elif nums[i] == nums[nums[i] - 1]:
            pass
        else:
            temp = 0
            temp = nums[nums[i] - 1]
            nums[nums[i] - 1] = nums[i]
            # if temp
            nums[i] = 0
        print (nums)
    out = 0
    for i, val in enumerate(nums):
        if val == 0:
            out = i+1
            break
    if out==0:
        out = max
    print (out)
    return out

# Driver code
# arr = [1, 2, 9,3, 1, 3, 6, 6,8,9,2]

# printRepeating(arr, arr_size)
# findDisappearedNumbers([1,1])
# firstMissingPositive([3,4,-1,1])
firstMIssingPositiveInteger([3,4,-1,1])#[0])#[1000,200,-1,1,7,2,1])