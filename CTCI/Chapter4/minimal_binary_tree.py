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

    # print('Node 2', root_node.val)
    left_val = minimal_tree(left_tree)
    print('left_val: ', left_val)
    root_node.left = left_val
    right_val = minimal_tree(right_tree)
    print('right_val: ', right_val)
    root_node.right = right_val

    return root_node



def factorial(n):
    if n == 1:
        return 1
    out =  n * factorial(n-1)
    return (out)

dict_ = {}
def fibonacci_memo(n):
    if n in dict_:
        return dict_[n]
    if n < 2:
        return n
    else:
        out = fibonacci_memo(n - 1) + fibonacci_memo(n - 2)
    dict_[n] = out
    return out


def fibonacci(n):
    if n < 2:
        return n
    out = fibonacci(n - 1) + fibonacci(n - 2)
    return out
    
    
def fibonacci_iter(n):
    if n<=2:
        return 1
    
    i = 1
    j = 1
    k = 0
    while k < n-2:
        val = i+j
        temp = i
        i = val
        j = temp
        
        k+=1
    return val
    

###################
# root_nodee = minimal_tree([5,6,7,8,9,10,11])
# print (root_nodee)

# print (factorial(4))

from time import time


start = time()
print(fibonacci(n=30))
print ('Simple Fibonacci: ', time()-start)


start = time()
print(fibonacci_memo(n=300))
print ('Memo Fibonacci: ',time()-start)

start = time()
print(fibonacci_iter(n=300))
print ('Memo Fibonacci: ',time()-start)


