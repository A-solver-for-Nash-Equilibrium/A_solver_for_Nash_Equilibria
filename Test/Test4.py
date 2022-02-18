# test cases for NESolver3/4

import os
import sys

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the path that contain NESolver package
sys.path.insert(0, lib_path)

from NESolver.NESolver5 import NESolver
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
    NE=NESol.analyze()
    print()
    print(NE)
    info = NESol.find()
    print()
    # print("This game has {len_} Nash Equilibrium: ".format(len_=info['NE_count'].sum()))
    print(info)


A = np.array([[8, 2, 2], [3, 9, 3], [-2, -2, 4]])
B = np.array([[1, 2, 4], [-2, 5, 4], [-2, 2, 7]])
n1 = ['I', 'J', 'F']
n2 = ['X','Y','Z']
# test_game("TTL2 b", A, B, n1, n2)
NESol = NESolver(A=A, B=B, action_name_1=n1, action_name_2=n2)
NESol.analyze()
info = NESol.find()
print(info)











