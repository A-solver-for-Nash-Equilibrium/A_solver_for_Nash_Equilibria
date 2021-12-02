# 2021/12/1

import numpy as np
import pandas as pd


class SDADeleter1:
    """
    Delete all the strictly dominated actions for a bi-matrix game
    SDA stands for strictly dominated actions
    """

    def __init__(self, A, B):
        """

        :param
            A: numpy array: payoff matric of player1
            B: dnumpy array: payoff matric of player2
        """
        # Turn array into dataframe to track action indices
        self.__A = pd.DataFrame(A)
        self.__B = pd.DataFrame(B)
        # number of player1&2's actions
        self.__m = A.shape[0]
        self.__n = B.shape[1]
        # delete all SDA and update A,B
        self.__recursively_delete_dominated_actions(player=1, first_turn=True, begin_of_one_turn=True)

    def __recursively_delete_dominated_actions(self, player, first_turn=False, begin_of_one_turn=False,
                                               outer_change=False):
        """
        Try to find and delete all strictly dominated actions of both players
        * will check dominated actions for row and column players repeatedly
                (checkA, checkB, checkA… each is called one turn, __find_one_strictly_dominated_action() may be called
                 more than once in one turn)
        * each player will be checked for at least once (param: first_turn)
        * the function returns when either player don’t have a dominated action in his/her turn
        * will change to check another player if no more dominated actions will be found in his/her turn
        :param
            player: int, 1 or 2
            first_turn: boolean, whether it's the first turn
            begin_of_one_turn: boolean, whether it's the first time to call __find_one_strictly_dominated_action()
                               in one turn
            outer_change: boolean, whether a SDA is found in one turn
        :return: None
        """
        inner_change = self.__find_one_strictly_dominated_action(player=player)
        if inner_change:
            if begin_of_one_turn:
                outer_change = True  # a SDA has been found in this turn
            # still check the old player until no SDA in this turn
            self.__recursively_delete_dominated_actions(player=player, outer_change=outer_change)
        else:
            # continue to check the other player if its the first turn or a SDA was found in this turn
            if first_turn or outer_change:
                new_player = 2 / player
                self.__recursively_delete_dominated_actions(player=new_player, begin_of_one_turn=True)
            # stop checking since no SDA in this turn, and its not the first turn
            else:
                return

    def __find_one_strictly_dominated_action(self, player):
        """
        Try to find and delete a strictly dominated action of the selected player
        Use dataframe here to track the index of actions
        This function will be called in the function 'iteratively_delete_dominated_actions()'
        :param player: int, 1 or 2
        :return: boolean, True if a SDA is found
        """
        if player == 1:
            target = self.__A
        else:
            target = self.__B.T

        n_rows = target.index.values  # row index
        # iteratively find whether ith action is a SDA
        for i in n_rows:
            for j in n_rows:
                if i != j:
                    if all(target.loc[i] < target.loc[j]):  # iloc->index from 0, loc-> index label
                        axis = player - 1  # player1: drop row, player2: drop column
                        self.__A.drop(i, axis=axis, inplace=True)
                        self.__B.drop(i, axis=axis, inplace=True)
                        return True
        return False

    def get_updated_payoff(self):
        return self.__A.values, self.__B.values

    def get_deleted_actions(self):
        deleted_m = []
        deleted_n = []
        n_deleted_actions = -1

        remaining_m = self.__A.index.values
        remaining_n = self.__B.columns.values

        for m in range(self.__m):
            if m not in remaining_m:
                deleted_m.append(m)
        for n in range(self.__n):
            if n not in remaining_n:
                deleted_n.append(n)

        n_deleted_actions = len(deleted_m + deleted_n)

        return n_deleted_actions, deleted_m, deleted_n

    def test(self):
        self.__find_one_strictly_dominated_action(2)
        print(self.__A)
        print(self.__B)


def main():
    def test_SDADeleter(A, B):
        print("player1's payoff:")
        print(A)
        print("player2's payoff:")
        print(B)
        A = pd.DataFrame(A)
        B = pd.DataFrame(B)
        SDA = SDADeleter1(A, B)
        n, dn, dm = SDA.get_deleted_actions()
        A, B = SDA.get_updated_payoff()
        print("{n} strictly dominated actions found, player1: {dn}, player2: {dm}".format(n=n, dn=dn, dm=dm))
        print("player1's new payoff matrix:")
        print(A)
        print("player2's new payoff matrix:")
        print(B)

    print('\n *****************')
    print([[(0, 2), ()]])
    M = np.array([[2, 2, 2], [4, 4, 4], [3, 3, 3]])
    test_SDADeleter(M, M)

    print('\n *****************')
    print([[(), (1, 2)],[(0,4),()]])
    M = np.array([[3, 2, 2, 2, 3], [4, 5, 5, 5, 4], [3, 3, 3, 3, 3]]).T
    test_SDADeleter(M, M)

    print('\n *****************')
    print([[(1), (0)]])
    M = np.array([[5, 7, 1], [1, 2, 3], [4, 5, 6]])
    test_SDADeleter(M, M)

    print('\n *****************')
    print([[(), (0)], [(0), ()]])  # ttl2
    A = np.array([[8, 2, 2], [3, 9, 3], [-2, -2, 4]])
    B = np.array([[1, 2, 4], [-2, 5, 4], [-2, 2, 7]])
    test_SDADeleter(A, B)

    print('\n *****************')
    print([[(2), (2)], [(0), (1, 2)]]) # https://www.youtube.com/watch?v=ErJNYh8ejSA
    A = np.array([[1, -1, 5, 1], [2, 1, 3, 5], [1, 0, 1, 0]])
    B = np.array([[1, 2, 0, 1], [3, 2, 0, 1], [1, 5, 7, 1]])
    test_SDADeleter(A, B)

    print('\n *****************')
    print([[(0), (2)], [(2), (0)], [(0, 2), ()]])  # exchange player1 and 2
    B = np.array([[1, -1, 5, 1], [2, 1, 3, 5], [1, 0, 1, 0]]).T
    A = np.array([[1, 2, 0, 1], [3, 2, 0, 1], [1, 5, 7, 1]]).T
    test_SDADeleter(A, B)


if __name__ == '__main__':
    main()
