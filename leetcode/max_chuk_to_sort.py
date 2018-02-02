



def max_chunk_to_sorted_distinct(arr):
    ans = ma = 0
    for i, x in enumerate(arr):
        ma = max(ma, x)
        print (i, ma, x)
        if ma == i:
            ans += 1
    print (ans)
    return ans


def max_chunk_to_sorted(arr):
    min_ = arr[0]
    max_ = arr[0]
    prev_min = arr[0]
    prev_max = arr[0]
    tot = 1
    for i in range(1,len(arr)):
        print (arr[i])
        print (prev_min, prev_max, min_, max_)
        if arr[i] < prev_min and arr[i]<prev_max and arr[i]<min_ and arr[i]<max_:
            tot = 1
        elif arr[i] > min_ and arr[i] > max_ and arr[i]>arr[i-1]:
            tot += 1
            prev_min = min_
            prev_max = max_
            min_ = min(min_, arr[i])
            max_ = max(max_, arr[i])

        elif arr[i]<prev_max:
            tot = max(tot-1, 1)
        print (tot)
    

        
max_chunk_to_sorted_distinct([2,4,3,1,5])#[0,4,5,2,1,3])#[5,4,3,2,1])#[0,2,1])#[3,1,7,5,6,2,0,8])#[1,0,2,3,4])#[5,4,3,2,1])#[1,2,3,4,5])