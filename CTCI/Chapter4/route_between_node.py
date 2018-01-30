from graphNode import Node, Graph

#
def createNewGraph():
    sizegraph = 12
    g = Graph(12)
    
    temp = [0] * sizegraph

    temp[0] = Node("a", 0)
    temp[1] = Node("b", 2)
    temp[2] = Node("c", 2)
    temp[3] = Node("d", 0)
    temp[4] = Node("e", 0)
    temp[5] = Node("f", 0)
    temp[6] = Node("1", 0)
    temp[7] = Node("2", 0)
    temp[8] = Node("3", 2)
    temp[9] = Node("4", 2)
    temp[10] = Node("5", 2)
    temp[11] = Node("6", 1)

    temp[1].addAdjacent(temp[0])
    temp[1].addAdjacent(temp[2])

    temp[2].addAdjacent(temp[1])
    temp[2].addAdjacent(temp[8])

    temp[8].addAdjacent(temp[4])
    temp[8].addAdjacent(temp[10])

    temp[9].addAdjacent(temp[6])
    temp[9].addAdjacent(temp[7])

    temp[10].addAdjacent(temp[9])
    temp[10].addAdjacent(temp[11])

    temp[11].addAdjacent(temp[5])

    for i in range(sizegraph):
        g.addNode(temp[i])
    return g


from collections import deque

def route_between_node_bfs(graph, start, end):
    print (graph.getNodes())
    que = deque()
    que.append(start)
    start.visited = True
    
    while len(que) >= 1:
        element = que[0]
        element.visited = True
        for adjecent_item in element.getAdjacent():
            if adjecent_item.visited == False:
                que.append(adjecent_item)
                if adjecent_item.getVertex() == end.getVertex():
                    return True
        que.popleft()

    return False

g = createNewGraph()
print (g)
n = g.getNodes()
print (n)
start = n[0]
end = n[10]
print ("Start at:", start.getVertex(), "End at: ", end.getVertex())

print (route_between_node_bfs(g, start, end))