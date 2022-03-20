# 2022/02/13
# move updating support to find()

import os
import sys

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the path that contain NESolver package
sys.path.insert(0, lib_path)

import gurobipy as gp
import numpy as np
from itertools import chain, combinations, product
import pandas as pd
from numpy import linalg
from SDADeleter.SDADeleter1 import SDADeleter1 as SDADeleter

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)


def powerset(iterable):
    """
    * To find all the subsets of the input value.
    * Refernce: https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    * Example usage: list(powerset("123")) --> [() (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)]
    * Time complexity : O(2^n), n is the length of the input
    :param
        iterable: ang iterable objective. In this case it's a list of numbers (action indices).
    :return
        A list of tuples. each tuple is a subset of the input iterable. The list contain all the subsets
            with an empty set at its beginning.
    """
    s = list(iterable)
    # change the range statement to range(1, len(s)+1) to avoid a 0-length combination
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def support_numeration(m, n):
    """
    * To calculate all the possible supports for a bi-matirx game.
    * The input is two lists of action index by default.
      The method will find all the combination between the subsets of the two list.
    * Time complexity (n is the number of actions of one player): O(4^n)
        - Use powerset() to calculate subsets: O(2^n)
        - Find the combination of the two subsets: O(2^n * 2^n) = O(2^(2n))
    :param
        m: list, list of player1's action indices
        n: list, list of player2's action indices
    :return
        support_m: list, player1's support
        support_n: list, player2's support
        support_all: list, all the supports of the two players, i.e. Cartesian product of support_m & support_n
    """
    # list of supports (by action index starting from 0) for player1: list of tuples
    support_m = list(powerset(m))
    support_m.pop(0)  # delete the empty one

    # same for player 2
    support_n = list(powerset(n))
    support_n.pop(0)

    # list of all supports for the game (Cartesian product): [((),()),()], list of 2-d tuples
    support_all = [s for s in product(support_m, support_n)]  # same as [(a,b) for a in ls1 for b in ls2]

    return support_m, support_n, support_all


def find_PNE(A, B):
    """
    * Find all PNE of bi-matrix game.
    * If a pure strategy with the payoff of player 1 being u1 and the payoff of player2 being u2 is a a Nash Equilibrum,
      then u1 should be the max value of the column that contains u1,
           u2 should be the max value of the row that contains u2.
    * Time complexity (assume each user has n actions): O(n^2)
        - Find the max value of one column or row: O(n)
        - Find all the position(s) that is a column max or row max: O(n^2)
        - Find all the position(s) that is both a column max of player1's payoff matrix and a row max of player2's
          payoff matrix, i.e. the intersection of column max and row max position: O(n)
    :param
        A: Pandas Dataframe or Numpy array, payoff matrix of player1.
        B: Pandas Dataframe or Numpy array, payoff matrix of player2.
    :return:
        PNE: List of tuples, each tuple is a PNE.
             (dataframe: original index; array: index from 0 for each player)
             e.g. [((1,),(2,)), ((2,),(1,))] means the game has 2 PNE.
                  One is that player1 chooses the action indexed as 1, player2 chooses the action indexed by 2.
                  One is that player1 chooses the action indexed as 2, player2 chooses the action indexed by 1.
    """
    A = pd.DataFrame(A)
    B = pd.DataFrame(B)
    # print(A)
    # print(B)

    # A: column max location, best response of player1 according to player2's action
    loc_A = set()
    # B: row mac location, best response of player2 according to player1's action
    loc_B = set()

    # find column max for player1
    col = A.columns.values
    for c in col:
        row = A[A[c] == A[c].max()].index.values
        for r in row:
            loc_A.add(((r,), (c,),))

    # find row max for player2
    B = B.T
    col = B.columns.values
    for c in col:
        row = B[B[c] == B[c].max()].index.values
        for r in row:
            loc_B.add(((c,), (r,),))

    # print(loc_A)
    # print(loc_B)

    PNE = sorted(loc_A.intersection(loc_B))
    # print(PNE)
    return PNE

def map_support(support, action_names):
    """
    * Map a list of supports with default index into the format of action name.
    * Worst case time complexity (assume each user has n actions, all supports admits NE):o(n * 2^n)
        - If we need to map all the possible supports from support numeration: each index will appear 2^(n-1) times,
          and we need to map all the 2 * n actions of 2 players: o(n * 2^n)
    :param
        support: A list of tuples, [((),()),...]
        action_names: A list of two lists of string, storing action names of two players, [[],[]]
    :return
        support_by_name: A list of tuples, [((),()),...], indices in support are translated to action names
    """
    support_by_name = []
    for s in support:
        # print(s)
        sbn_1 = []
        sbn_2 = []
        for i in s[0]:
            sbn_1.append(action_names[0][i])
        for j in s[1]:
            sbn_2.append(action_names[1][j])
        sbn = (tuple(sbn_1),tuple(sbn_2))
        support_by_name.append(sbn)
    return support_by_name

"""
settings for NESolver
"""
eps = 0.00001  # epsilon for strict inequality
precision = '.6f'


class NESolver:
    """
    NESolver to find all the Nash Equilibria of a bi-matrix game
    """

    def __init__(self, A, B, action_name_1=None, action_name_2=None):  # A,B are numpy arrays, payoff matrix
        """
        check input values and initialized
        :param
            A: 2-d numpy array, player1's payoff matrix
            B: 2-d numpy array, player2's payoff matrix
            action_name_1: list, player1's action names
            action_name_2: list, player2's action names
        """

        # check input
        if not (isinstance(A, np.ndarray) and isinstance(B, np.ndarray)):
            raise TypeError("The input two payoff matrices should be numpy array.")
            sys.exit(1)
        if A.shape != B.shape:
            raise ValueError("The input two payoff matrices should have same size.")
            sys.exit(1)
        if action_name_1:
            if not isinstance(action_name_1, list):
                raise TypeError("The input action names should be two list.")
                sys.exit(1)
            if len(action_name_1) != A.shape[0]:
                raise ValueError("The length of input action names should match the number of actions.")
                sys.exit(1)
        if action_name_2:
            if not isinstance(action_name_2, list):
                raise TypeError("The input action names should be two list.")
                sys.exit(1)
            if len(action_name_2) != B.shape[1]:
                raise ValueError("The length of input action names should match the number of actions.")
                sys.exit(1)

        # initialize
        self.__A = A
        self.__B = B

        # default action name by index from 0
        if not action_name_1:
            action_name_1 = list(range(A.shape[0]))
        if not action_name_2:
            action_name_2 = list(range(B.shape[1]))
        self.__action_names = [action_name_1, action_name_2]

        self.__lp_1_ini, self.__lp_2_ini = self.__init_lp_model()

        # will be updated in self.__find()
        self.__support = ()
        self.__NE_log_dict = {}
        # will be updated in self.__analysis()
        self.__NE_info = {}


    def __init_lp_model(self):
        """
        initialize and return two basic LP models, each for a player, same for all the supports
          x,y --> strategy vector for player 1 and 2
          A* = A.dot(x), B* = B.T.dot(y)
          w1, w2 --> expected payoff of player1 and 2
        :return
            lpm_1.copy(): gurobi model
                variables: x, w2, B*
                constraints: sum x = 1, x >= 0, B* = B.T * x
            lpm_2.copy(): gurobi model
                variables: y, w1, A*
                constraints: sum y = 1, y >= 0, A* = A * y
        """
        # number of actions of player1 & 2 respectively
        m = self.__A.shape[0]
        n = self.__B.shape[1]

        lpm_1 = gp.Model("lpm_1")  # x, w2, B*
        lpm_2 = gp.Model("lpm_2")  # y, w1, A*

        # suppress optimization information from gurobi
        lpm_1.Params.LogToConsole = 0
        lpm_2.Params.LogToConsole = 0

        # strategy
        x = lpm_1.addMVar(shape=m, name='x')
        lpm_1.addConstrs((x[i] >= 0 for i in range(m)))  # no need, '>= 0' is default gurobi setting
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
        * Here, x (player 1) is related to payoff matrix B, y(player 2) is related to payoff matrix A.
        * Take player2 as an example
          - Trim A so that it only contains row and columns that represent actions in the support
          - Form matrix M based on A:
                1 1 ... 1 0
                a a ... a -1
                a a ... a -1
                . . ... .  .
                a a ... a -1
          - Form matrix b: [1 0 0 ... 0].T
          - If rank(M) = rank(M|b) < the number of M's columns, there are infinite many NE.
        * Time complexity:
            Trim A, Rank(M|b)
        :param
            player: int, 1 or 2
            support: list of two lists, each showing the support of one player
        :return
            infinity: boolean, true if the player have infinit NE under the support
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
        To find Nash Equilibrium of a given support.
        * Update contraints based on the support, take player2 as an example:
            - If action i in support:
                - yi > 0
                - w1 = A*i
            - else:
                - yi = 0
                - w1 >= A*1
        * Optimize the two LP through gurobi
        * If a LP has a feasible solution, check whether the player will have infinitely many NE.
        * Record all the information into NE_log_dict.
        :param
            support: tuple of two tuples ((),()), each is a support of a player
        :return:
            NE_log_dict: dict, computation result based on the support
               {'x_count': 0 or 1 or inf, Number of player1's strategy that may form a NE;
                'x_value': None or tuple, An example of player1's strategy;
                'y_count': 0 or 1 or inf, Number of player2's strategy that may form a NE;
                'y_value': None or tuple, An example of player2's strategy;
                'NE_count': 0 or 1 or inf, Number of NE;
                'NE_value': None or list of two tuples, An example of NE;
                'w1': None or float, Player1's expected payoff;
                'w2': None or float, Player2's expected payoff;
                'NE_type': -1 or PNE or MNE }
                (NE_count is 0 if any of x_count, y_count is 0)
        """
        # create a dict with default value to store computing results
        NE_log_dict = {'x_count': 0, 'x_value': None, 'y_count': 0, 'y_value': None, 'NE_count': 0,
                       'NE_value': None, 'w1': None, 'w2': None, 'NE_type': -1}

        # number of actions of each player
        m = self.__A.shape[0]
        n = self.__B.shape[1]

        # initialize lpsolver for each support
        lpm_1 = self.__lp_1_ini.copy()  # x, w2, B*
        lpm_2 = self.__lp_2_ini.copy()  # y, w1, A*
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

        # check whether the support admit NE, update the number of NE and NE_log_dict
        # same for both players
        try:
            lpm_1.optimize()
            p = tuple(format(lpm_1.getVarByName('x[{_i}]'.format(_i=i)).x, precision) for i in range(m))
        except Exception as err:
            pass
        else:
            # print(lpm_1.getVars())
            NE_log_dict['w2'] = lpm_1.getVarByName('w2').x
            NE_log_dict['x_value'] = p
            if self.__check_NE_infinity(player=1, support=support):  #
                NE_log_dict['x_count'] = float('inf')
            else:
                NE_log_dict['x_count'] = 1

        try:
            lpm_2.optimize()
            q = tuple(format(lpm_2.getVarByName('y[{_j}]'.format(_j=j)).x, precision) for j in range(n))
        except Exception as err:
            pass
        else:
            # print(lpm_2.getVars())
            NE_log_dict['w1'] = lpm_2.getVarByName('w1').x
            NE_log_dict['y_value'] = q
            if self.__check_NE_infinity(player=2, support=support):
                NE_log_dict['y_count'] = float('inf')
            else:
                NE_log_dict['y_count'] = 1

        # update NE info if this support has NE
        if NE_log_dict['x_count'] != 0 and NE_log_dict['y_count'] != 0:
            # update NE_value
            NE_log_dict['NE_value'] = [p, q]
            # update NE_count
            if NE_log_dict['x_count'] == float('inf') or NE_log_dict['y_count'] == float('inf'):
                NE_log_dict['NE_count'] = float('inf')
            else:  # (NE_count: 0 / inf / 1)
                NE_log_dict['NE_count'] = 1
            # update NE_type
            if len(support[0]) + len(support[1]) == 2:
                NE_log_dict['NE_type'] = 'PNE'
            else:
                NE_log_dict['NE_type'] = 'MNE'
        return NE_log_dict

    def __find_NE(self):
        """
        Compute NE of all the support regardless of strictly dominated actions
        You can see clearly here which player fails to admit a NE via the value of w1 and w2
        Assign value to self.__NE_log_dict.
        """
        # support numeration
        self.__support = support_numeration(range(self.__A.shape[0]), range(self.__B.shape[1]))[2]
        # record the results of all support into self.__NE_log_dict
        for support in self.__support:
            self.__NE_log_dict[support] = self.__compute_equilibrium_of_one_support(support)

    def find(self):
        """
        Calculate self.__NE_log_dict if it's empty.
        : return
            NE_log_df: pandas dataframe
                same content as in self.__NE_log_dict, store the computation result for each support
        """
        # update if it's empty
        if not self.__NE_log_dict:
            self.__find_NE()

        # convert dict into dataframe for an easy checking
        NE_log_df = pd.DataFrame({'support': map_support(self.__NE_log_dict.keys(),self.__action_names),
                                  'NE_type': [self.__NE_log_dict[k]['NE_type'] for k in self.__NE_log_dict],
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

    def __analyze_NE(self):
        """
        Similar to self.__find, this method also finds all the NE of a given game, but through: SDA -> PNE -> MNE.
        * Delete all the strictly dominated actions from payoff matrices via class SDADeleter
        * Find PNE based on updated payoff matrices via method find_PNE (column & row max)
        * Do support numeration without strictly dominated actions, delete support with two pure strategy, and delete
            support with one-pure strategy if no PNE
        * Find mixed NE based on support numeration
        * Record results in to self.__NE_info.
        """
        # list of action indices of each player, will delete strictly dominated actions later
        # m_actions = range(self.__A.shape[0])
        # n_actions = range(self.__B.shape[1])

        # check the existence of strictly dominated actions, update payoff matrices
        SDA = SDADeleter(self.__A, self.__B)
        n_SDA, deleted_indices_1, deleted_indices_2 = SDA.get_deleted_actions()
        new_A, new_B = SDA.get_updated_payoff()  # dataframe

        # find PNE based on updated payoff matrix
        PNE = find_PNE(new_A, new_B)  # list
        n_PNE = len(PNE)

        # do support numeration without SDA
        support = support_numeration(new_A.index.values, new_B.columns.values)[2]
        # if n_PNE == 0:  # if no PNE, then neither player will have pure strategy in MNE
        #     new_support = [s for s in support if len(s[0]) > 1 and len(s[1]) > 1]
        # else:  # delete PNE
        #     new_support = [s for s in support if (len(s[0]) + len(s[1])) > 2]
        new_support = [s for s in support if (len(s[0]) + len(s[1])) > 2]

        # find mixed NE based on support numeration
        n_MNE = 0
        MNE = {}  # dict
        if (len(new_support) != 0):
            for s in new_support:
                info = self.__compute_equilibrium_of_one_support(s)
                if info['NE_count'] != 0:  # only record NE
                    MNE[s] = {}
                    MNE[s]['NE_count'] = info['NE_count']
                    MNE[s]['NE_value'] = info['NE_value']
                    n_MNE += info['NE_count']

        # update self.__NE_info
        self.__NE_info['n_SDA'] = n_SDA
        self.__NE_info['player1_SDA'] = [self.__action_names[0][i] for i in deleted_indices_1]
        self.__NE_info['player2_SDA'] = [self.__action_names[1][i] for i in deleted_indices_1]
        self.__NE_info['n_PNE'] = n_PNE
        self.__NE_info['PNE'] = PNE
        self.__NE_info['n_MNE'] = n_MNE
        self.__NE_info['MNE'] = MNE


    def analyze(self, show_info=1):
        """
        Compute self.__NE_info if it is empty.
        Print out information on SDA, PNE, MNE.
        :return
            self.__NE_info: dict
                {'n_SDA': int, number of strictly dominated actions,
                 'player1_SDA': list, player1's strictly dominated actions,
                 'player2_SDA': list, player2's strictly dominated actions,
                 'n_PNE': int, number of pure Nash Equilibria
                 'PNE': List of tuples, each tuple is a PNE.
                 'n_MNE': int, number of mixed Nash Equilibria
                 'MNE': dict, records the number and value of NE for each support (dict key) that admits mixed NE. }
        """
        # update if empty
        if not self.__NE_info:
            self.__analyze_NE()

        # convert MNE into dataframe for printing
        MNE_df = pd.DataFrame()
        MNE = self.__NE_info['MNE']
        if MNE:  # if MNE is not empty
            MNE_df = pd.DataFrame({'support': map_support(MNE.keys(),self.__action_names),
                                   'NE_count': [MNE[k]['NE_count'] for k in MNE],
                                   'NE_value': [MNE[k]['NE_value'] for k in MNE]
                                   })

        # change dict MNE to MNE_df in return value, change indices in support to action name
        NE_info = self.__NE_info
        NE_info['MNE'] = MNE_df
        NE_info['PNE'] = map_support(self.__NE_info['PNE'], self.__action_names)

        if show_info == 1:
            print('======= Analyze NE =======')
            print('n_SDA:\t{}'.format(NE_info['n_SDA']))
            print('1_SDA:\t{}'.format(NE_info['player1_SDA']))
            print('2_SDA:\t{}'.format(NE_info['player2_SDA']))
            print('---------------')
            print('n_PNE:\t{}'.format(NE_info['n_PNE']))
            if NE_info['n_PNE'] == 0:
                print('PNE:\tNone')
            else:
                print('PNE:')
                for pne in NE_info['PNE']:
                    print('\t{}'.format(pne))
            # print('PNE:\t{}'.format(self.__NE_info['PNE']))
            print('---------------')
            print('n_MNE:\t{}'.format(NE_info['n_MNE']))
            if NE_info['n_MNE'] == 0:
                print('MNE:\tNone')
            else:
                print('MNE:')
                print(NE_info['MNE'])

        return self.__NE_info

    def test(self):
        self.find()
        for i in self.__NE_log_dict:
            print(i)
            print(self.__NE_log_dict[i])


def main():
    A = np.array([[2, 0], [0, 1]])
    # A = [[2, 0], [0, 1]]
    # A = np.array([[0], [2]])
    B = np.array([[1, 0], [0, 2]])
    # B = [[2, 0], [0, 1]]
    # B = np.array([[0], [2]])
    a = ['a','s']
    # a = (2,2)
    # a = [1,2,3]
    b = ['b1','b2']
    # b = (1,2,3)
    # b = [1,2,3]
    NE = NESolver(A, B,a,b)
    NE.analyze()


if __name__ == '__main__':
    main()
