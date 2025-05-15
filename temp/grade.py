import ast
from cognitive_complexity.api import get_cognitive_complexity

code = """
def min_cost(cost, m, n):
    tc = [[0 for x in range(C)] for x in range(R)]
    tc[0][0] = cost[0][0]
    for i in range(1, m+1):
        tc[i][0] = tc[i-1][0] + cost[i][0]
    for j in range(1, n+1):
        tc[0][j] = tc[0][j-1] + cost[0][j]
    for i in range(1, m+1):
        for j in range(1, n+1):
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]
    return tc[m][n]
"""

code1 = """
def decimal_to_binary(n):
    return bin(n).replace("0b","")
"""
code2 = """
import sys
def next_smallest_palindrome(num):
    numstr = str(num)
    for i in range(num+1,sys.maxsize):
        if str(i) == str(i)[::-1]:
            return i
"""

tree = ast.parse(code2)
funcdef = tree.body[0]  # Get the function definition
complexity = get_cognitive_complexity(funcdef)
print(f"Cognitive Complexity: {complexity}")