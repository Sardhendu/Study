



arr = [7,5,1,3,5,4,3,6,2, 1]



def coursera_1(arr):
    j = 1
    sum = 0
    non_arr = []
    for i in range(0,len(arr)):
        flag = 0
        while j<=len(arr)-1:
            if arr[j] < arr[i]:
                sum += arr[i] - arr[j]
                flag = 1
                break
            j+=1
        if flag == 0:
            print ('asdaasdasd ', i, j)
            sum += arr[i]
            non_arr.append(i)
        j = i+1
        print (sum)
        print (non_arr)
    # return sum, non_arr

# sum +=


def coursera_problem(a):
    sum = 0
    elem_pos = []
    for i in range(len(a)):
        j = 1
        while j < len(a) - i:
            if a[i] > a[i + 1]:
                sum += a[i] - a[i + 1]
            else:
                if a[i] > min(a[i + 1:]):
                    sum += a[i] - min(a[i + 1:])
                else:
                    sum += a[i]
                    elem_pos.append(i)
            j += 1
            break
    sum += a[i]
    elem_pos.append(i)
    print (sum)
    print (elem_pos)
#     # return sum, elem_pos
#
#
# coursera_1(arr)

# coursera_problem(arr)


from collections import deque

def coursera_problem_On(arr):
    arr2 = [None]*10
    arr2_idx = [None]*10
    temp = arr[0]
    temp_idx = 0
    j = 0
    arr2[j] = arr[0]
    arr2_idx[j] = 0
    for i in range(1,len(arr)):
        print (arr[i], arr2[j], i, arr2_idx[j])
        if arr[i]< arr2[j] and i> arr2_idx[j]:
            arr2[j] = arr[i]
            arr2_idx[j] = i
            if i!=len(arr)-1:
                temp = arr[i+1]
        elif arr[i] > arr2[j] and i> arr2_idx[j]:
            if arr[i] == min(temp, arr[i]):
                j+=1
                arr2[j] = arr[i]
                arr2_idx[j] = i
                temp = min(temp, arr[i])
        print (temp)
        print (arr2)
        print (arr2_idx)
    
    print('')
    print('')
    
    
    sum = 0
    j = 0
    for i, val in enumerate(arr):
        if i ==  arr2_idx[j]:
            sum += val
        elif val > arr[i+1]:
            sum+=val-arr[i+1]
        elif arr2_idx[j] > i :
            sum += val - arr[arr2_idx[j]]
    
        print (sum)
        if i ==  arr2_idx[j]:
            j+=1
    
    print ('final sum')
    # sum += val
    print (sum)
 
 
arr = [7,5,1,3,5,4,2,6,7,4,3]
arr = [7,5,1,3,1,4,2,6,7,4,3]

arr = [5,1,3,4,6,2]
coursera_problem_On(arr)