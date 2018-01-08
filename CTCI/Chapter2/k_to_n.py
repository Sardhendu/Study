from LinkedList import LinkedList

def kidx_to_n(inp, kth_idx):
    curr = inp.head
    for i in range(0,kth_idx):
        if not curr:
            return None
        else:
            curr = curr.next
    inp.head = curr
    return inp


def kval_to_n(inp, kth_val):
    curr = inp.head
    while curr:
        if curr.value == kth_val:
            break
        else:
            curr = curr.next
    inp.head = curr
    return inp
    
    
    
random_num = LinkedList().generate(10, 1, 100)
print (random_num)
print (kidx_to_n(random_num, 4))
print ('')
random_num = LinkedList().generate(20, 1, 5)
print (random_num)
print (kval_to_n(random_num, 4))
