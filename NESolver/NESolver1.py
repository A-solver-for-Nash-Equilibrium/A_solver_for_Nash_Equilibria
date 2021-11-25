import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import chain, combinations, product

"""
If extend to >= 3 player: 
input payoff --> [n-dimensional], best response --> M*[][player index], strategy --> p[][player index]
"""


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


class NESolver1:
    """
    NESolver to find all the Nash Equilibria of a bi-matrix game
    """

    def __init__(self, A, B):  # A,B are numpy arrays, payoff matrix
        self.__A = A
        self.__B = B
        self.__NE = []
        self.__initial_lp = self.__init_lp_model()
        self.__support = self.__support_numeration()[2]

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
        return a basic LP model, same for all the supports
        x,y --> strategy vector for player 1 and 2
        A* = A.dot(x), B* = B.T.dot(y)
        w1, w2 --> expected payoff for player1 and 2
        """
        m = self.__A.shape[0]  # number of player1's actions
        n = self.__B.shape[1]  # number of player2's actions

        lpm = gp.Model("lpm")
        lpm.Params.LogToConsole = 0  # suppress optimization information from gurobi

        # strategy
        x = lpm.addMVar(shape=m, name='x')
        lpm.addConstrs((x[i] >= 0 for i in range(m)))  # no need, default gurobi settings
        lpm.addConstr(x.sum() == 1)
        y = lpm.addMVar(shape=n, name='y')
        lpm.addConstrs((y[j] >= 0 for j in range(n)))
        lpm.addConstr(y.sum() == 1)

        # expected payoff
        w1 = lpm.addVar(lb=-float('inf'), name="w1")
        w2 = lpm.addVar(lb=-float('inf'), name="w2")

        # for compute best response actions
        A_ = lpm.addMVar(lb=-float('inf'), shape=m, name='A*')  # A y
        lpm.addConstr(A_ - self.__A @ y == 0)
        B_ = lpm.addMVar(lb=-float('inf'), shape=n, name='B*')  # B^T x
        lpm.addConstr(B_ - self.__B.T @ x == 0)

        # objective functions
        lpm.setObjective(0)  # ? cannot show whether x,y have more than one answer
        # lpm.setObjective(np.array([1, 1]) @ x, GRB.MAXIMIZE)

        lpm.update()
        # print(lpm.getVars())
        return lpm.copy()

    def __compute_equilibrium_of_one_support(self, support):
        """
        To find equilibrium of a given support
        Will add the equilibrium in to self.__NE if find one
        :param
            support: tuple of two tuples ((),()), each is a support for a player
        :return:
            exist: boolean; whether the given support admit a nash equilibrium
            equilibrium: None if exist == False.
                         Otherwise, A list of two tuples [(),()], each is a strategy for a player.
        """

        # number of actions each player
        m = self.__A.shape[0]
        n = self.__B.shape[1]

        lpm = self.__initial_lp.copy()
        # print(lpm.getVars())
        w1 = lpm.getVarByName('w1')
        w2 = lpm.getVarByName('w2')

        support_a = support[0]  # one possible support of player 1, tuple
        e = sys.float_info.epsilon
        e = 0.00001
        for i in range(m):
            if i in support_a:
                lpm.addConstr(lpm.getVarByName('x[{_i}]'.format(_i=i)) >= e)
                lpm.addConstr(w1 - lpm.getVarByName('A*[{_i}]'.format(_i=i)) == 0)
            else:
                lpm.addConstr(w1 - lpm.getVarByName('A*[{_i}]'.format(_i=i)) >= 0)
                lpm.addConstr(lpm.getVarByName('x[{_i}]'.format(_i=i)) == 0)

        support_b = support[1]  # one possible support of player 1, tuple
        for j in range(n):
            if j in support_b:
                lpm.addConstr(lpm.getVarByName('y[{_j}]'.format(_j=j)) >= e)
                lpm.addConstr(w2 - lpm.getVarByName('B*[{_j}]'.format(_j=j)) == 0)
            else:
                lpm.addConstr(w2 - lpm.getVarByName('B*[{_j}]'.format(_j=j)) >= 0)
                lpm.addConstr(lpm.getVarByName('y[{_j}]'.format(_j=j)) == 0)

        exist = False  # whether the support admit a equilibrium
        equilibrium = None
        try:
            lpm.optimize()  # ? how to change to fractions, here .4f
            p = tuple(format(lpm.getVarByName('x[{_i}]'.format(_i=i)).x, '.4f') for i in range(m))
            q = tuple(format(lpm.getVarByName('y[{_j}]'.format(_j=j)).x, '.4f') for j in range(n))
        except Exception as err:
            pass
            # print("support " + str(support)+ "  does not admit a single Nash Equilibrium")
            # print("Oops. " + str(err) + ", this support does not admit a single Nash Equilibrium")
        else:
            exist = True
            equilibrium = [p, q]
            self.__NE.append(equilibrium)
        finally:
            pass
            # print("finish solving a LP")

        return exist, equilibrium

    def find(self):
        for support in self.__support:
            self.__compute_equilibrium_of_one_support(support)
        return self.__NE

    def test(self):
        i=0
        for support in self.__support:
            i+=1
            print(i)
            print(support)
            print(self.__compute_equilibrium_of_one_support(support)[1])


def main():
    A = np.array([[2, 2], [2, 2]])
    B = np.array([[3, 3], [3, 4]])
    NESolver = NESolver1(A,B)
    NESolver.test()

if __name__ == '__main__':
    main()
