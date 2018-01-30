
#
#
# def fibonacci(n):
#     '''
#     :return:
#     '''
#     print (''n)
#     if n==0:
#         return 0
#
#     if n==1:
#         return 1
#
#     return fibonacci(n-1) + fibonacci(n-2)
#
# fibonacci(n=5)


#
arr = [1,2,3,4,5,6,7,8]
n = len(arr) - 1
k = 7


str_min = len(arr)
i =0
j = 0
temp = 0
break_a = 0


def exchange(a, b, arr):
    tot = arr[a] + arr[b]
    arr[a] = tot - arr[a]
    arr[b] = tot - arr[b]
    return arr
    
    
while True:
    print('Running ', break_a)
    print ('temp, and i ', temp, i)
    
    if i <= n-k:
        a = i
        b = k+i

        print('index repl ', a, b)
        print('val_repl', arr[a], arr[b])
        
        if break_a != 0 and temp == b:
            print('yesyesyesyeyes')
            i = temp + 1
            temp = i
            break_a += 1
            print ('NEW i and temp', i, temp)
            continue
            
        arr = exchange(a, b, arr)
        i = k + i
        sub_ind = k+i
        print ('Value ', arr)
        print ('New i is: ', i)
        print ('')
        
    else:
        print('index repl ', i, k + i - n - 1)
        print('val_repl', arr[i], arr[k + i - n - 1])
        a = i
        b = k + i - n - 1

        if break_a != 0 and temp == b:
            print('yesyesyesyeyes')
            i = temp + 1
            temp = i
            break_a += 1
            print('NEW i and temp', i, temp)
            continue

        arr = exchange(a, b, arr)
        i = k+i-n-1
        print('Value ', arr)
        print('New i is: ', i)
        print ('')

    break_a += 1
    
    if break_a == 7:
        break
        
    
    
        # tot = arr[i] + arr[k+i]
        # arr[i] = tot - arr[i]
        # arr[k+i] = tot - arr[k+i]
        
        
# rem = n+1-i
# print (str_min)
# print ('lenleft ', rem)



#
# arr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# n = len(arr) - 1
# k = 4
# str_min = len(arr)
# for i in range(0,len(arr)):
#     print (i)
#     if i <= n-k:
#         print('index repl ', i, k+i)
#         print ('val_repl', arr[i], arr[k+i])
#         tot = arr[i] + arr[k+i]
#         arr[i] = tot - arr[i]
#         arr[k+i] = tot - arr[k+i]
#         if str_min>k+i:
#             str_min = k+i
#         print ('Value ', arr)
#         print ()
#     else:
#         break
# rem = n+1-i
# print (str_min)
# print ('lenleft ', rem)
#
#
# if k>n/2:
#     left = rem - (n + 1 - k)
#     right = rem - rem + (n + 1 - k)
#     print('left, right ',left ,right )
#     for j in range(i, len(arr)):
#         print ('input j is ', j)
#         if j + left <len(arr):
#             print (arr[j],  arr[j + left])
#             tot = arr[j] + arr[j + left]
#             arr[j] = tot - arr[j]
#             arr[j + left] = tot - arr[j + left]
#             print('Value ', arr)
#             print()
#         else:
#             print ('deducting left')
#             left = left/2
#             print(arr[j], arr[j + left])
#             tot = arr[j] + arr[j + left]
#             arr[j] = tot - arr[j]
#             arr[j + left] = tot - arr[j + left]
#             print('Value ', arr)
#             print()
#     print (arr)
#

# set1 = k-i
# set2 = n-k + 1
# print(set1, set2)
# for
    # else:
    #     print('Entering part 2')
    #     print ('index repl', i, k+i-n-1)
    #     print ('val_repl', arr[i], arr[k+i-n-1])
    #     tot = arr[i] + arr[k+i-n-1]
    #     arr[i] = tot - arr[i]
    #     arr[k+i-n-1] = tot - arr[k+i-n-1]
    #
    #     print('Value ', arr)
    #     print()