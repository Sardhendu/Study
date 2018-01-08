from LinkedList import LinkedList

def check_if_palindrome(list1):
    len_list1 = len(list1)
    if len_list1%2 == 0:
        stop_idx = len_list1 // 2 + len_list1 % 2
    else:
        stop_idx = len_list1//2
    
    print (stop_idx)
    curr1 = list1.head
    list2 = LinkedList()
    
    for i in range(0, stop_idx):
        list2.add_to_beginning(curr1)
        curr1 = curr1.next
        list1.head = curr1


    if len_list1 % 2 != 0:
        curr1 = curr1.next
        list1.head = curr1
    
    curr1 = list1.head
    curr2 = list2.head
    
    flag = True
    for i in range(0,stop_idx):
        
        if str(curr1) != str(curr2):
            print('Thes two values dont match: ', curr1.value, curr2.value)
            flag = False
            break
        curr1 = curr1.next
        curr2 = curr2.next
    return flag
    


######## 1 
list1 = LinkedList()
list1.add_multiple([1,6,1,7,7,1,6,1])
print (check_if_palindrome(list1))
