

arr = [1,2,3,5,7,8,9]

i = 0
k = 0
j = len(arr) - 1
l = len(arr) - 1

tot = 10
temp = arr[i]
while k < tot/2:
    print (k,l,i,j, temp, arr[j])
    if j <= i:
        k += 1
        i = k
    
    if temp+arr[j] == tot:
        print ('yupyupyupyup: ', arr[k:i+1], arr[j])
        temp = temp + arr[i+1]
        i += 1
        j -= 1
    elif temp+arr[j] > tot:
        j -= 1
    elif temp+arr[j] < tot:
        temp = temp + arr[i+1]
        i += 1
        j = l
        
    
        
        
