

def computeArea(A,B,C,D,E,F,G,H):
    over_lap = max((min(C,G) - max(A, E)), 0) * max((min(D, H) - max(B,F)), 0)
    
    return (C-A)*(B-D) + (E-G)*(F-H) - over_lap
    
    
# computeArea(A=5,B=6,C=7,D=8,E=6,F=5,G=8,H=7)


arr_out = []
path = '/Users/sam/All-Program/App-DataSet/intern_min_path/small_triangle.txt'
def minimal_path_traversal():
    with open(path) as f:
        for num, line in enumerate(f):
            line_arr = line.split('  ')
            if num == 0:
                print(int(line_arr[0]))
                idx = 0
            else:
                if int(line_arr[idx]) < int(line_arr[idx+1]):
                    print (line_arr[idx])
                    idx = idx
                else:
                    print (line_arr[idx+1])
                    idx = idx+1
minimal_path_traversal()