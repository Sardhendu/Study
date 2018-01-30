# from graphNode import Graph, Node

class Node():
    '''
        The parent node would always be an object of the class node
    '''
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right= None

    # nicely printable string representation of an object
    def __str__(self):
        print ('1111111111111111')
        print (self.left, self.val, self.right)
        return '%s-%s-%s'%(self.left, self.val, self.right)


out_arr = []
def minimal_tree(inp_arr):
    if len(inp_arr) == 0:
        print('############################')
        return ''

    mid = len(inp_arr) //2
    root_node = Node(inp_arr[mid])
    left_tree = inp_arr[0:mid]
    right_tree = inp_arr[mid+1:]

    print('Node ', root_node.val)
    print ('Left Tree ',len(left_tree), left_tree)
    print ('Right Tree ', len(right_tree), right_tree)

    print('Node 2', root_node.val)
    left_val = minimal_tree(left_tree)
    print('left_val: ', left_val)
    root_node.left = left_val
    right_val = minimal_tree(right_tree)
    print('right_val: ', right_val)
    root_node.right = right_val

    return root_node

root_nodee = minimal_tree([5,6,7,8,9,10,11])
print (root_nodee)




