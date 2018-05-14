# python3

import sys, threading
sys.setrecursionlimit(10**7) # max depth of recursion
threading.stack_size(2**27)  # new thread will get stack of such size

class TreeHeight:
        def read(self):
                # self.n = int(sys.stdin.readline())
                # self.parent = list(map(int, sys.stdin.readline().split()))
                self.n = 5
                self.parent = [4, -1, 4, 1, 1]
            
        # def compute_height(self):
        #         # Replace this code with a faster implementation
        #         maxHeight = 0
        #         for vertex in range(self.n):
        #                 print ('vertex ', vertex)
        #                 height = 0
        #                 i = vertex
        #                 print('i before while ', i)
        #                 while i != -1:
        #                         height += 1
        #                         i = self.parent[i]
        #                         print ('i in while ', i)
        #                 maxHeight = max(maxHeight, height);
        #         return maxHeight;
        
        def compute_height(self):
                max_val = 0
                for vertex in range(self.n):
                        max(max_val, )
        

def main():
  tree = TreeHeight()
  tree.read()
  print(tree.compute_height())

threading.Thread(target=main).start()
