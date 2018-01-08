
from LinkedList import LinkedList, LinkedListNode

def sum_list(list1, list2):
    if len(list1) < len(list2):
        extra = len(list2) - len(list1)
        for i in range(0, extra):
            list1.add(0)
    else:
        extra = len(list1) - len(list2)
        for i in range(0, extra):
            list2.add(0)
            
    curr1 = list1.head
    curr2 = list2.head
    list3 = LinkedList()
    i = 0
    carry = 0
    while curr1:
        nw_val = curr1.value + curr2.value + carry
        if nw_val>=10:
            list3.add(nw_val%10)
            carry = int(nw_val/10)
        else:
            list3.add(nw_val)
            carry = 0

        print (list3)
        i += 1
        curr1 = curr1.next
        curr2 = curr2.next
        
    if carry:
        list3.add(carry)
    return list3


def sum_list_followup(list1, list2):
    if len(list1) < len(list2):
        extra = len(list2) - len(list1)
        for i in range(0, extra):
            list1.add_to_beginning(0)
    else:
        extra = len(list1) - len(list2)
        for i in range(0, extra):
            list2.add_to_beginning(0)
    
    
    curr1 = list1.head
    curr2 = list2.head
    list3 = LinkedList()
    
    iter1 = 0
    len_buff = len(list1)
    buff = 0
    while curr1:
        buff = buff + (curr1.value * pow(10, len_buff-1)) +  (curr2.value * pow(10, len_buff-1))
        curr1 = curr1.next
        curr2 = curr2.next
        len_buff -= 1
    
    list3.add_multiple([i for i in str(buff)])
    return list3
        
    

######## 1
list1 = LinkedList()
list1.add_multiple([7, 1, 6,1])

list2 = LinkedList()
list2.add_multiple([5, 9, 2])

print (sum_list(list1, list2))


######## 2
list1 = LinkedList()
list1.add_multiple([6, 1, 7])

list2 = LinkedList()
list2.add_multiple([2, 9, 5])

print (sum_list_followup(list1, list2))