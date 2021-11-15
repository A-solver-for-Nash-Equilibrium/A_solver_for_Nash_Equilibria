import os
import sys
lib_path = os.path.dirname(os.path.abspath(__file__))  # the path that contain NESolver package
sys.path.insert(0, lib_path)
from NESolver.NESolver1 import NESolver1
import numpy as np

""" Example:
Enter the number of row player's actions:2
Enter the number of column player's actions:2
Enter the 2 x 2 payoff matrix of row player:
separate elements by typing space, separate rows by typing enter
2 0
3 1
Enter the 2 x 2 payoff matrix of column player:
separate elements by typing space, separate rows by typing enter
2 3
0 1
player 1's payoff matrix:
[[2 0]
 [3 1]]
player 2's payoff matrix:
[[2 3]
 [0 1]]
Restricted license - for non-production use only - expires 2022-01-13
This game has 1 Nash Equilibrium: 
[('0.0000', '1.0000'), ('0.0000', '1.0000')]
"""
rows = int(input("Enter the number of row player's actions:"))
columns = int(input("Enter the number of column player's actions:"))

A = []
print("Enter the %s x %s payoff matrix of row player:" % (rows, columns))
print("separate elements by typing space, separate rows by typing enter")
for i in range(rows):
    A.append(list(map(int, input().rstrip().split())))

B = []
print("Enter the %s x %s payoff matrix of column player:" % (rows, columns))
print("separate elements by typing space, separate rows by typing enter")
for i in range(rows):
    B.append(list(map(int, input().rstrip().split())))

A = np.array(A)
B = np.array(B)

print("player 1's payoff matrix:")
print(A)
print("player 2's payoff matrix:")
print(B)
NESolver = NESolver1(A=A, B=B)
NE = NESolver.find()
print("This game has {_len} Nash Equilibrium: ".format(_len=len(NE)))
for e in NE:
    print(e)