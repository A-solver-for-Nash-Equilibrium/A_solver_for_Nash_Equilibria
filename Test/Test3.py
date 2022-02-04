# test cases for NESolver3/4

import os
import sys

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the path that contain NESolver package
sys.path.insert(0, lib_path)

from NESolver.NESolver4 import NESolver4 as NESolver
import numpy as np
import pandas as pd


def test_game(game_name, A, B, n1=None, n2=None):
    print()
    print("*************************")
    print(game_name)

    m = "player 1's actions: {}"
    if n1:
        print(m.format(n1))
    else:
        print(m.format('default index from 0'))
    n = "player 2's actions: {}"
    if n2:
        print(n.format(n2))
    else:
        print(n.format('default index from 0'))

    print("player 1's payoff matrix:")
    print(A)
    print("player 2's payoff matrix:")
    print(B)

    NESol = NESolver(A=A, B=B, action_name_1=n1, action_name_2=n2)
    NESol.analyze()
    # NE = NESol.find()
    # print()
    # print("This game has {len_} Nash Equilibrium: ".format(len_=NE['NE_count'].sum()))
    # print(NE)



# examples in lecture 1

A = np.array([[2, 0], [3, 1]])
B = np.array([[2, 3], [0, 1]])
n = ['Quiet', 'Fink']
test_game("The Prisoner's Dilemma", A, B, n, n)

A = np.array([[1, -1], [-1, 1]])
B = np.array([[-1, 1], [1, -1]])
n = ['Head', 'Tail']
test_game("Matching Pennies", A, B, n, n)

A = np.array([[2, 0], [0, 1]])
B = np.array([[1, 0], [0, 2]])
n = ['Bach', 'Stravinsky']
test_game("Bach or Stravinsky?", A, B, n, n)

# examples in TTL 2

A = np.array([[7, 2, 3], [2, 7, 4]])
B = np.array([[2, 7, 6], [7, 2, 5]])
n1 = ['T', 'B']
n2 = ['L', 'M', 'R']
test_game("TTL2 a", A, B, n1, n2)

A = np.array([[8, 2, 2], [3, 9, 3], [-2, -2, 4]])
B = np.array([[1, 2, 4], [-2, 5, 4], [-2, 2, 7]])
n = ['I', 'J', 'F']
test_game("TTL2 b", A, B, n, n)

# examples in lecture 2

A = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
B = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
n = ['R', 'S', 'P']
test_game("rock-scissors-paper", A, B, n, n)

A = np.array([[6, 1, 2], [1, 6, 3]])
B = np.array([[1, 6, 5], [6, 1, 4]])
n1 = ['U', 'D']
n2 = ['L', 'M', 'R']
test_game("lec2 P59", A, B, n1, n2)

# self-designed

A = np.array([[2, 2], [2, 2]])
B = np.array([[3, 3], [3, 4]])
test_game("self-designed 1", A, B)

A = np.array([[1, 1]])
B = np.array([[2, 2]])
test_game("self-designed 2", A, B)

A = np.array([[1, -1, 5, 1], [2, 1, 3, 5], [1, 0, 1, 0]])
B = np.array([[1, 2, 0, 1], [3, 2, 0, 1], [1, 5, 7, 1]])
test_game("https://www.youtube.com/watch?v=ErJNYh8ejSA", A, B)

A = np.array([[3,1], [0,2]])
B = np.array([[2,1], [0,3]])
n=['Baseball', 'Ballet']
test_game('https://saylordotorg.github.io/text_introduction-to-economic-analysis/s17-03-mixed-strategies.html',A,B,n,n)

A = np.array([[2,0], [1,3]])
B = np.array([[1,2], [2,0]])
test_game('http://faculty.smu.edu/ozerturk/MixedNash.pdf',A,B, ['T','B'],['L','R'])

A = np.array([[-10,10,-1], [0,-1,-1], [4,-1,2]])
B = np.array([[4,0,-1], [10,-1,1], [-10,-1,2]])
n1 = ['L','M','R']
n2 = ['l','m','r']
test_game('https://math.stackexchange.com/questions/853107/finding-mixed-nash-equlibrium', A, B,n1,n2)

# examples in ttl10
A = np.array([[0,6], [3,3], [2,5]])
B = np.array([[3,1], [3,2], [2,6]])
n1 = ['T','M','B']
n2 = ['L', 'R']
test_game('ttl10 1', A, B,n1,n2)

# final 27
A = np.array([[9,5], [5,9]])
B = np.array([[5,9], [9,5]])
n1 = ['a','b']
n2 = ['c', 'd']
test_game('final 27', A, B,n1,n2)

# final 37
A = np.array([[-2,8], [0,4]])
B = np.array([[-2,0], [8,4]])
n1 = ['a','b']
n2 = ['c', 'd']
test_game('final 37', A, B,n1,n2)









