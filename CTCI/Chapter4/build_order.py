from graphNode import Graph, Node
from collections import deque

# from sets import set
def build_order(arr, dependency_list_of_tuple):
    adj_length_dict = {}
    for i, j in dependency_list_of_tuple:
        try:
            adj_length_dict[str(i)] += 1
        except KeyError:
            adj_length_dict[str(i)] = 1
    # print (adj_length_dict)
    myDict = {}
    temp = [0]*(len(arr)-1)  # -1 because one of the arr element in not in the dependency list
    ind = 0
    for i,j in dependency_list_of_tuple:
        # print ('')
        # print ('startstartstart ', i,j, list(myDict.keys()))
        if i not in list(myDict.keys()) and j not in list(myDict.keys()):
            # print('i and j not in list ', ind, i, j)
            temp[ind] = Node(str(i), adj_length_dict[str(i)])
            myDict[str(i)] = ind
            # print ('vertex', temp[ind].getVertex())
            ind+=1
            
            try:
                temp[ind] = Node(str(j), adj_length_dict[str(j)])
                # print('vertex', temp[ind].getVertex())
            except KeyError:
                temp[ind] = Node(str(j), 0)
            myDict[str(j)] = ind
            temp[ind - 1].addAdjacent(temp[ind])
            ind+=1
            
            # print (temp)
            # print('adjacent', temp[ind-2].getAdjacent())
            
        elif i not in list(myDict.keys()) and j in list(myDict.keys()):
            # print('j in list ', ind, str(j), myDict[str(j)])
            myDict[str(i)] = ind
            temp[ind] = Node(str(i), adj_length_dict[str(i)])
            myDict[str(i)] = ind
            ind += 1
            temp[ind - 1].addAdjacent(temp[myDict[j]])
            
        elif i in list(myDict.keys()) and j not in list(myDict.keys()):
            # print('i in list ', ind, str(i), myDict[str(i)])
            myDict[str(j)] = ind
            # print('vertex', temp[myDict[str(i)]].getVertex())
            temp[ind] = Node(str(j), 0)
            
            temp[myDict[i]].addAdjacent(temp[ind])
            ind += 1
        else:
            # print('both in list ', ind, str(i), str(j), myDict[str(i)], myDict[str(j)])
            temp[myDict[i]].addAdjacent(temp[myDict[j]])
    # print (temp)
    
    # FIND THE ROOT NODE (THAT HAS HIGH)
    
    root_str = list(adj_length_dict.keys())[list(adj_length_dict.values()).index(max(adj_length_dict.values()))]
    root_node_onb_pos = myDict[str(root_str)]
    root_node = temp[root_node_onb_pos]
    
    # print (root_node)
    
    outlist = []
    queue = deque()
    queue.append(root_node)
    
    mySet = set()
    while len(queue) >= 1:
        if queue[0].getVertex() not in mySet:
            outlist.append(queue[0].getVertex())
            mySet.add(queue[0].getVertex())

        for i in queue[0].getAdjacent():
            queue.append(i)
        queue.popleft()
    # print (outlist)
    return outlist
    
dependency_list_of_tuple = [('a','d'),('f','b'),('b','d'),('f','a'),('d','c')]
print (build_order(['a','b','c','d','e','f'],dependency_list_of_tuple))