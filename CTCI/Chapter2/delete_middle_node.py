from LinkedList import LinkedList

# search and delete middle
def del_middle_node(inp, middle_node):
    middle_node.value = middle_node.next
    middle_node.next = middle_node.next.next
    return inp


inp = LinkedList()
inp.add_multiple([1, 2, 3, 4])
middle_node = inp.add(5)
print (middle_node)
inp.add_multiple([7, 8, 9])

print (inp)
print (del_middle_node(inp, middle_node))
# print (inp)