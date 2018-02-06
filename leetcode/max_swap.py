

def maximumSwap(num):
    a = list(str(num))
    for i in range(0, len(a) - 1):
        num_max = int(a[i])
        # print ('new_num_max: ', num_max)
        max_j = 0
        for j in range(i+1, len(a)):
            # print (int(a[i]) != int(a[j]))
            if int(a[i]) != int(a[j]):
                # print ('ij:', i,j)
                num_max = max(num_max, int(a[j]))
                if num_max == int(a[j]):
                    max_j = j
                    # print('max_j, num_max: ', max_j, num_max)
        if max_j != 0:
            # print ('time to swap')
            a[i] = str(int(a[i]) + int(a[max_j]))
            a[max_j] = str(int(a[i]) - int(a[max_j]))
            a[i] = str(int(a[i]) - int(a[max_j]))
            break
    return int(''.join([i for i in a]))
    
print(maximumSwap(97987897987))