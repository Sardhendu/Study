def urlify(string):
    string_list = list(string)
    print (string_list)
    
    st_len = len(string.strip())
    tot_len = len(string_list)
    
    
    for i in reversed(range(st_len)):
        print (i)
        if string_list[i] == ' ':
            string_list[tot_len - 3 : tot_len] = '%20'
            tot_len = tot_len-3
        else:
            string_list[tot_len - 1 ] = string_list[i]
            tot_len = tot_len-1
    return ('').join(string_list)



def one_away(st1, st2):
    st1 = st1.lower()
    st2 = st2.lower()
    st1_arr = list(st1)
    st2_arr = list(st2)
    flag_not = True
    if len(st1) == len(st2):
        flag = 0
        for i, st in enumerate(st1_arr):
            if st != st2_arr[i]:
                flag += 1
            if flag > 1:
                flag_not = False
                break
    
    elif abs(len(st1) - len(st2)) == 1:
        j = 0
        flag = 0
        
        for i, st in enumerate(st1_arr):
            
            if i == len(st1) - 1:
                if flag > 1 :
                    flag_not = False
                    break
                else:
                    break
            print(st, st2_arr[j], i, j, flag)
            if st == st2_arr[j]:
                j += 1
            else:
                flag += 1
                
                
            if flag > 1:
                flag_not = False
                break
    else:
        flag_not = False
    return flag_not
        
    
def string_compress(string):
    st_arr = list(string.lower())
    sm_arr = []
    
    count = 0
    prev_st = string[0]
    sm_arr.append(prev_st)
    for st in st_arr[1:len(string)]:
        if st == prev_st:
            count += 1
        else:
            if count == 0:
                sm_arr.append(st)
            else:
                sm_arr.append(count)
                sm_arr.append(st)
            count = 0
    print (sm_arr)
    return ('').join(sm_arr)
    

import numpy as np





def isSibstring(st1, st2):
    st1 = list(st1)
    st2 = list(st2)
    s2_idx = 0
    s1_arr = []
    for s1_idx, s1 in enumerate(st1):
        # print(s1_idx, s1, s2_idx, st2[s1_idx])
        if s1 == st2[s2_idx]:
            s2_idx += 1
        else:
            s1_arr = st1[0:s1_idx+1]
            print (s1_arr)
            st1 = st1[s1_idx+1:len(st1)] + s1_arr
            return st1
    return 'Yes'

def string_rotation(st1, st2):
    i = 0
    flag = False
    while i != len(st1) - 1:
        st1 = isSibstring(st1, st2)
        if st1 == 'Yes':
            flag = True
            break
        i += 1
    return flag



    
# print(urlify('my name is sam      '))
print(one_away('ppl', 'ap'))

# print (string_compress(string='aaaabbcdeffffaaaabd'))
# print (rotmat)



# print (string_rotation('abcdeabc', 'abcabcde'))



    


