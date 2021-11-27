# 2021/11/24

# try to solver the infinite NE problem
#   if the LP has solution, check the rank of coefficients to see the number of solutions

# separately find the value for X and Y

# There are many code duplicates here due to processing player 1 and 2 are quite similar.
# The differences are in the strings: '{x[i_]}', '{y[i_]}', 'x_value','y_value'...
# I'll try combining some code into methods later.

import sys
from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import chain, combinations, product
import pandas as pd
from numpy import linalg

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

eps = 0.00001  # epsilon for strict inequlity
precision = '.6f'

# log_column = ['x_count', 'x_value', 'y_count', 'y_value', 'NE_count', 'NE_value', 'w1', 'w2']
log_column_dict = {'x_count': 0, 'x_value': None, 'y_count': 0, 'y_value': None, 'NE_count': 0, 'NE_value': None,
                   'w1': None, 'w2': None}


def powerset(iterable):
    """
    https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    Haven't understand underlined logic
    returns a iterator?
    list(powerset("abcd")) --> [() (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)]
    """
    s = list(iterable)
    # change the range statement to range(1, len(s)+1) to avoid a 0-length combination
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class NESolver2:
    """
    NESolver to find all the Nash Equilibria of a bi-matrix game
    """

    def __init__(self, A, B):  # A,B are numpy arrays, payoff matrix
        self.__A = A
        self.__B = B
        # self.__NE = []
        self.__lp_1_ini, self.__lp_2_ini = self.__init_lp_model()
        self.__support = self.__support_numeration()[2]
        # self.__NE_log_df = pd.DataFrame(index=self.__support, columns=log_column) # easy to check
        # self.__NE_log_dict = dict.fromkeys(self.__support, log_column_dict) # same value address for all keys
        self.__NE_log_dict = self.init_NE_log_dict()

    def init_NE_log_dict(self):  # should be conducted after self.__support ..
        NE_log_dict={}
        for support in self.__support:
            NE_log_dict[support] = deepcopy(log_column_dict)
        return NE_log_dict

    def __support_numeration(self):
        # list of supports (by action index starting from 0) for player1: list of tuples
        support_a = list(powerset(np.arange(self.__A.shape[0])))
        support_a.pop(0)  # delete the empty one

        # same for player 2
        support_b = list(powerset(np.arange(self.__B.shape[1])))
        support_b.pop(0)

        # list of all supports for the game (Cartesian product): [((),()),()], list of 2-d tuples
        support_all = [s for s in product(support_a, support_b)]  # same as [(a,b) for a in ls1 for b in ls2]

        return support_a, support_b, support_all

    def __init_lp_model(self):
        """
        return two basic LP models, each for a player, same for all the supports
        x,y --> strategy vector for player 1 and 2
        A* = A.dot(x), B* = B.T.dot(y)
        w1, w2 --> expected payoff for player1 and 2
        """
        m = self.__A.shape[0]  # number of player1's actions
        n = self.__B.shape[1]  # number of player2's actions

        lpm_1 = gp.Model("lpm_1") # x, w2, B*
        lpm_2 = gp.Model("lpm_2") # y, w1, A*
        lpm_1.Params.LogToConsole = 0  # suppress optimization information from gurobi
        lpm_2.Params.LogToConsole = 0

        # strategy
        x = lpm_1.addMVar(shape=m, name='x')
        lpm_1.addConstrs((x[i] >= 0 for i in range(m)))  # no need, default gurobi settings
        lpm_1.addConstr(x.sum() == 1)
        y = lpm_2.addMVar(shape=n, name='y')
        lpm_2.addConstrs((y[j] >= 0 for j in range(n)))
        lpm_2.addConstr(y.sum() == 1)

        # expected payoff
        w1 = lpm_2.addVar(lb=-float('inf'), name="w1")
        w2 = lpm_1.addVar(lb=-float('inf'), name="w2")

        # for compute best response actions
        A_ = lpm_2.addMVar(lb=-float('inf'), shape=m, name='A*')  # A y
        lpm_2.addConstr(A_ - self.__A @ y == 0)
        B_ = lpm_1.addMVar(lb=-float('inf'), shape=n, name='B*')  # B^T x
        lpm_1.addConstr(B_ - self.__B.T @ x == 0)

        # objective functions
        lpm_1.setObjective(0)
        lpm_2.setObjective(0)

        lpm_1.update()
        lpm_2.update()
        # print(lpm.getVars())
        return lpm_1.copy(), lpm_2.copy()

    def __check_NE_infinity(self, player, support):
        """
        Check whether the player will have infinitely many NE under the support
        Here, x (player 1) is realted to payoff matrix B, y(player 2) is related to payoff matrix A
        :param player: int, 1 or 2
        :param support:
        :return: infinity: boolean
        """
        infinity = False

        if player == 1:
            M = self.__B
        else:
            M = self.__A

        # delete unneeded coefficients
        M = M[list(support[0]), :][:, list(support[1])]
        if player == 1:
            M = M.T

        # change to the form M c = b, c=[x1, x2,..., w2].T or [y1, y2,..., w1].T
        top = np.ones(shape=(1, M.shape[1]))
        M = np.vstack((top, M))
        right = - np.ones(shape=(M.shape[0], 1))
        M = np.hstack((M, right))
        M[0, -1] = 0

        # augmented matrix [M|b]
        b = np.zeros(shape=(M.shape[0], 1))
        b[0, 0] = 1
        Mb = np.hstack((M, b))

        # check whether there are infinitely many solutions
        n = M.shape[1]
        rank_M = linalg.matrix_rank(M)
        rank_Mb = linalg.matrix_rank(Mb)
        if rank_M == rank_Mb and rank_M < n:
            infinity = True

        return infinity

    def __compute_equilibrium_of_one_support(self, support):
        """
        To find equilibrium of a given support
        Will update self.__NE_log_dict[support]
        :param
            support: tuple of two tuples ((),()), each is a support for a player
        :return:
            {support: self.__NE_log_dict[support]}
        """

        # number of actions of each player
        m = self.__A.shape[0]
        n = self.__B.shape[1]

        # initialize lpsolver for each support
        lpm_1 = self.__lp_1_ini.copy() # x, w2, B*
        lpm_2 = self.__lp_2_ini.copy() # y, w1, A*
        # print(lpm_1.getVars())
        w1 = lpm_2.getVarByName('w1')
        w2 = lpm_1.getVarByName('w2')

        # epsilon for x>0, y>0, since gurobi doesn't allow strict inequality
        e = eps

        # update lp according to support
        # same for both players
        support_a = support[0]  # one possible support of player 1, tuple
        for i in range(m):
            if i in support_a:
                lpm_1.addConstr(lpm_1.getVarByName('x[{_i}]'.format(_i=i)) >= e)
                lpm_2.addConstr(w1 - lpm_2.getVarByName('A*[{_i}]'.format(_i=i)) == 0)
            else:
                lpm_1.addConstr(lpm_1.getVarByName('x[{_i}]'.format(_i=i)) == 0)
                lpm_2.addConstr(w1 - lpm_2.getVarByName('A*[{_i}]'.format(_i=i)) >= 0)

        support_b = support[1]  # one possible support of player 2, tuple
        for j in range(n):
            if j in support_b:
                lpm_2.addConstr(lpm_2.getVarByName('y[{_j}]'.format(_j=j)) >= e)
                lpm_1.addConstr(w2 - lpm_1.getVarByName('B*[{_j}]'.format(_j=j)) == 0)
            else:
                lpm_2.addConstr(lpm_2.getVarByName('y[{_j}]'.format(_j=j)) == 0)
                lpm_1.addConstr(w2 - lpm_1.getVarByName('B*[{_j}]'.format(_j=j)) >= 0)

        # check whether the support admit NE, the number of NE, update NE_log_dict
        # same for both players
        try:
            lpm_1.optimize()
            p = tuple(format(lpm_1.getVarByName('x[{_i}]'.format(_i=i)).x, precision) for i in range(m))
        except Exception as err:
            pass
            # self.__NE_log_dict[support]['x_count'] = 0 # default value
            # self.__NE_log_dict[support]['x_value'] = None # default value
        else:
            # print(lpm_1.getVars())
            self.__NE_log_dict[support]['w2'] = lpm_1.getVarByName('w2').x
            self.__NE_log_dict[support]['x_value'] = p
            if self.__check_NE_infinity(player=1, support=support): #
                self.__NE_log_dict[support]['x_count'] = float('inf')
            else:
                self.__NE_log_dict[support]['x_count'] = 1

        try:
            lpm_2.optimize()
            q = tuple(format(lpm_2.getVarByName('y[{_j}]'.format(_j=j)).x, precision) for j in range(n))
        except Exception as err:
            pass
            # self.__NE_log_dict[support]['y_count'] = 0 # default value
            # self.__NE_log_dict[support]['y_value'] = None # default value
        else:
            # print(lpm_2.getVars())
            self.__NE_log_dict[support]['w1'] = lpm_2.getVarByName('w1').x
            self.__NE_log_dict[support]['y_value'] = q
            if self.__check_NE_infinity(player=2, support=support):
                self.__NE_log_dict[support]['y_count'] = float('inf')
            else:
                self.__NE_log_dict[support]['y_count'] = 1

        # update NE info
        if self.__NE_log_dict[support]['x_count'] != 0 and self.__NE_log_dict[support]['y_count'] != 0:
            self.__NE_log_dict[support]['NE_value'] = [p, q]
            if self.__NE_log_dict[support]['x_count'] == float('inf') or self.__NE_log_dict[support]['y_count'] == float('inf'):
                self.__NE_log_dict[support]['NE_count'] = float('inf')
            if self.__NE_log_dict[support]['x_count'] == 1 and self.__NE_log_dict[support]['y_count'] == 1:
                self.__NE_log_dict[support]['NE_count'] = 1

        return {support: self.__NE_log_dict[support]}

    def find(self):
        # compute all the support, log NE into self.__NE_log_dict
        for support in self.__support:
            self.__compute_equilibrium_of_one_support(support)

        # convert dict into dataframe for an easy checking
        NE_log_df = pd.DataFrame({'support': self.__NE_log_dict.keys(),
                                  'NE_count': [self.__NE_log_dict[k]['NE_count'] for k in self.__NE_log_dict],
                                  'NE_value': [self.__NE_log_dict[k]['NE_value'] for k in self.__NE_log_dict],
                                  'w1': [self.__NE_log_dict[k]['w1'] for k in self.__NE_log_dict],
                                  'w2': [self.__NE_log_dict[k]['w2'] for k in self.__NE_log_dict],
                                  'x_value': [self.__NE_log_dict[k]['x_value'] for k in self.__NE_log_dict],
                                  'x_count': [self.__NE_log_dict[k]['x_count'] for k in self.__NE_log_dict],
                                  'y_value': [self.__NE_log_dict[k]['y_value'] for k in self.__NE_log_dict],
                                  'y_count': [self.__NE_log_dict[k]['y_count'] for k in self.__NE_log_dict],
                                  })
        return NE_log_df

    def test(self):
        self.find()
        for i in self.__NE_log_dict:
            print(i)
            print(self.__NE_log_dict[i])



def main():
    A = np.array([[-2, 1.5], [1.5, -2]])
    B = np.array([[-2, 1.5], [1.5, -2]])
    NESolver = NESolver2(A, B)
    NE=NESolver.find()
    print("player 1's payoff matrix:")
    print(A)
    print("player 2's payoff matrix:")
    print(B)
    print("This game has {_len} Nash Equilibrium: ".format(_len=NE['NE_count'].sum()))
    print(NE)
    # NESolver.test()

if __name__ == '__main__':
    main()
