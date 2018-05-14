# python3

import sys

class Bracket:
    def __init__(self, bracket_type, position):
        self.bracket_type = bracket_type
        self.position = position

    def Match(self, c):
        if self.bracket_type == '[' and c == ']':
            return True
        if self.bracket_type == '{' and c == '}':
            return True
        if self.bracket_type == '(' and c == ')':
            return True
        return False

if __name__ == "__main__":
    text = sys.stdin.read()
    # text = 'foo(bar[i);'
    # text = 'foo(bar);'
    # text = 'a{'
    # text = 'foo(bar[i];'
    opening_brackets_stack = []
    out = 'Success'
    for i, next in enumerate(text):
        if next == '(' or next == '[' or next == '{':
            # Process opening bracket, write your code here
            opening_brackets_stack.append(next)

        if next == ')' or next == ']' or next == '}':
            # Process closing bracket, write your code here
            if len(opening_brackets_stack) == 0:
                out = i+1
                break
            else:
                obj_b = Bracket(opening_brackets_stack[-1], i)
                # print ('aaaaaa', opening_brackets_stack[-1])
                if not obj_b.Match(next):
                    out = i+1
                opening_brackets_stack.pop()
    if len(opening_brackets_stack) !=0:
        out = i+1
    print (out)
    # Printing answer, write your code here
