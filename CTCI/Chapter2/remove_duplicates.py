from LinkedList import LinkedList
    
# Using python list
def rmv_duplicates_follow_up(inp):
    i = 0
    curr = inp[i]
    while curr:
        j = i+1
        nxt = inp[j]
        k = 0
        while nxt:
            if curr == nxt:
                inp.pop(j)
                k += 1
            else:
                j += 1
            
            if j  == len(inp):
                break
            else:
                nxt = inp[j]
        i += 1
        
        if i == len(inp):
            break
        else:
            curr = inp[i]
    return inp


# Using LinkedList

def rmv_duplicates_follow_up_ll(inp):
    print (inp.head)
    curr = inp.head
    while curr:
        compare = curr
        while compare.next:
            if curr.value == compare.next.value:
                compare.next = compare.next.next
            else:
                compare = compare.next
        curr = curr.next
    return inp



                                      
# a = list(np.random.randint(1, 10, 100))
# print (rmv_duplicates_follow_up(a))

#
random_num = LinkedList().generate(10, 1, 5)
print (random_num)
print (rmv_duplicates_follow_up_ll(random_num))
            