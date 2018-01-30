from graphNode import Graph, Node
from collections import deque

# For this question lets assume the binary tree is a basically a graph,
# The logic remains the same even if it is a tree or a graph

def create_balanced_binary_graph_tree():
    sizegraph = 16
    g = Graph(16)
    
    temp = [0] * sizegraph
    
    temp[0] = Node(8, 2)
    
    temp[1] = Node(6, 2)
    temp[2] = Node(10, 2)
    
    temp[3] = Node(5, 2)
    temp[4] = Node(7, 2)
    temp[5] = Node(9, 2)
    temp[6] = Node(11, 2)
    
    temp[7] = Node(3, 1)
    temp[8] = Node(5.5, 0)
    temp[9] = Node(6.5, 0)
    temp[10] = Node(7.5, 0)
    temp[11] = Node(8.5, 0)
    temp[12] = Node(9.5, 0)
    temp[13] = Node(10.5, 0)
    temp[14] = Node(12, 0)

    temp[15] = Node(2, 0)

    
    temp[0].addAdjacent(temp[1])
    temp[0].addAdjacent(temp[2])
    
    temp[1].addAdjacent(temp[3])
    temp[1].addAdjacent(temp[4])
    
    temp[2].addAdjacent(temp[5])
    temp[2].addAdjacent(temp[6])
    
    temp[3].addAdjacent(temp[7])
    temp[3].addAdjacent(temp[8])

    temp[4].addAdjacent(temp[9])
    temp[4].addAdjacent(temp[10])

    temp[5].addAdjacent(temp[11])
    temp[5].addAdjacent(temp[12])

    temp[6].addAdjacent(temp[13])
    temp[6].addAdjacent(temp[14])

    temp[7].addAdjacent(temp[15])
    
    for i in range(sizegraph):
        g.addNode(temp[i])
    return g


def create_unbalanced_binary_graph_tree():
    sizegraph = 9
    g = Graph(9)
    
    temp = [0] * sizegraph
    
    temp[0] = Node(8, 2)
    
    temp[1] = Node(6, 2)
    temp[2] = Node(10, 0)
    
    temp[3] = Node(5, 2)
    temp[4] = Node(7, 1)
    
    temp[5] = Node(3, 1)
    temp[6] = Node(5.5, 0)
    temp[7] = Node(7.5, 0)
    
    temp[8] = Node(2, 0)
    
    temp[0].addAdjacent(temp[1])
    temp[0].addAdjacent(temp[2])
    
    temp[1].addAdjacent(temp[3])
    temp[1].addAdjacent(temp[4])
    
    temp[3].addAdjacent(temp[5])
    temp[3].addAdjacent(temp[6])
    
    temp[4].addAdjacent(temp[7])
    
    temp[5].addAdjacent(temp[8])
    
    
    for i in range(sizegraph):
        g.addNode(temp[i])
    return g



# FOR BOTH BALANCED AND UNBALANCED BINARY TREE
def list_of_depths_balanced_tree (graph, element):
    queue = deque()
    # Add the first element
    queue.append(element)
    list = []
    list.append(element.getVertex())

    depth = 0
    assume_node = 0
    while len(queue) >= 1:
        print (assume_node, depth)
        assume_node += 1
        parent = queue[0]
        if pow(2, depth) == assume_node:
            print(list)
            depth += 1
            list = []
        queue.popleft()
        
        list1 = []
        for adj_node in parent.getAdjacent():
            list.append(adj_node.getVertex())
            queue.append(adj_node)
    print(list)

# # When balanced tree
# def list_of_depths_unbalanced_tree(graph, element):
#     queue = deque()
#     # Add the first element
#     queue.append(element)
#     list = []
#     list.append(element.getVertex())
#     print (list)
#     depth = 2
#     assume_node = 2
#     list = []
#     while len(queue) >= 1:
#
#         parent = queue[0]
#         queue.popleft()
#
#         adj_nodes = parent.getAdjacent()
#         print ('parent', parent.getVertex())
#         print('len of adj nodes ', len(adj_nodes))
#         assume_node += 2-len(adj_nodes)  # If there are no adjecent nodes then assume node would
#                                          # increment by 2, if there is only 1 then it would increment by 1
#
#         for adj_node in adj_nodes:
#             assume_node += 1
#             list.append(adj_node.getVertex())
#             queue.append(adj_node)
#
#         print('depth ', depth, 'assume_node', assume_node)
#
#         if pow(2, depth) == assume_node:
#             print('@@@@@@@####### ', list)
#             depth += 1
#             list = []
#     print(list)


###########  RUN
graph = create_balanced_binary_graph_tree()
nodes = graph.getNodes()
root = nodes[0]
print ('The root is: ', root)
list_of_depths_balanced_tree(graph, root)