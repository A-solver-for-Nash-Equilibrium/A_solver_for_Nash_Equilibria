# test cases for NESolver2

import os
import sys
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the path that contain NESolver package
sys.path.insert(0, lib_path)

from NESolver.NESolver2 import NESolver2
import numpy as np
import pandas as pd

def test_game(game_name, A, B):
    print()
    print("*************************")
    print(game_name)
    print("player 1's payoff matrix:")
    print(A)
    print("player 2's payoff matrix:")
    print(B)
    NESolver = NESolver2(A=A, B=B)
    NE = NESolver.find()
    print("This game has {_len} Nash Equilibrium: ".format(_len=NE['NE_count'].sum()))
    print(NE)


# examples in lecture 1

A = np.array([[2, 0], [3, 1]])
B = np.array([[2, 3], [0, 1]])
test_game("The Prisoner's Dilemma", A, B)

A = np.array([[1, -1], [-1, 1]])
B = np.array([[-1, 1], [1, -1]])
test_game("Matching Pennies", A, B)

A = np.array([[2, 0], [0, 1]])
B = np.array([[1, 0], [0, 2]])
test_game("Bach or Stravinsky?", A, B)

# examples in TTL 2

A = np.array([[7, 2, 3], [2, 7, 4]])
B = np.array([[2, 7, 6], [7, 2, 5]])
test_game("TTL2 a", A, B)

A = np.array([[8, 2, 2], [3, 9, 3], [-2, -2, 4]])
B = np.array([[1, 2, 4], [-2, 5, 4], [-2, 2, 7]])
test_game("TTL2 b", A, B)

# examples in lecture 2

A = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
B = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
test_game("rock-scissors-paper", A, B)

A = np.array([[6, 1, 2], [1, 6, 3]])
B = np.array([[1, 6, 5], [6, 1, 4]])
test_game("lec2 P59", A, B)

# self-designed

A = np.array([[2, 2], [2, 2]])
B = np.array([[3, 3], [3, 4]])
test_game("self-designed 1", A, B)

A = np.array([[1, 1]])
B = np.array([[2, 2]])
test_game("self-designed 2", A, B)
