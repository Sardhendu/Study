


def search(nums, target):
    j = 0
    i = 0
    # k = 0
    counter = 0
    while i<len(nums):
        print (nums)
        if nums[0] == target[0]:
            # print (nums[0], target[0])
            counter += 1
        else:
            nums.append(nums[0])
            nums.pop(0)
        i+=1
    if nums == target:
        return True
    else:
        return False

        
        
# print (search([0, 1, 2, 4, 5, 6, 7], [4, 5, 6, 7, 0, 1, 2]))

print (search([], 5))