# Introduction

The project computes all pure and mixed Nash Equilibria of a bi-matrix game.

Supported precision :  0.00001

Main steps:

* Delete strictly dominated actions.
* Find all PNE based on column max and row max.
* Do support numeration, and calculate mixed NE based on each support using gurobi.

# File structure

* NESolver
  * NESolver4.py
* SDADeleter
  * SDADeleter1.py
* Test
  * Test3.py

# Software Dependencies

```
python 			3.8.12
gurobipy 		9.5.0
numpy 			1.21.2
pandas			1.3.4
itertools
os
sys
```

# Example Usage

```python
from NESolver.NESolver4 import NESolver4 as NESolver
import numpy as np

A = np.array([[8, 2, 2], [3, 9, 3], [-2, -2, 4]])
B = np.array([[1, 2, 4], [-2, 5, 4], [-2, 2, 7]])
n1 = ['I', 'J', 'F']
n2 = ['X','Y','Z']

NESol = NESolver(A=A, B=B, action_name_1=n1, action_name_2=n2)
NESol.analyze()
info = NESol.find()
print(info)
```

## Input

***class NESolver(A, B, action_name_1=None, action_name_2=None)***


* A :  2-d numpy array, payoff matrix of player 1

  B :  2-d numpy array, payoff matrix of player 2
  
  ```python
  A = np.array([[8, 2, 2], [3, 9, 3], [-2, -2, 4]])
  B = np.array([[1, 2, 4], [-2, 5, 4], [-2, 2, 7]])
  ```

* action_name_1 :  list (optional), action names of player1

  action_name_2 :  list (optional), action names of player2

  (Actions will be indexed from 0 if action_name_1, action_name_2 are None.)
  
  ```python
  n1 = ['I', 'J', 'F']
  n2 = ['X', 'Y', 'Z']
  ```

* A, B should have the same size. The length of a, b should match the shape of A or B.

* Will raise TypeError if input is of wrong datatype :

  ```
  TypeError: The input two payoff matrices should be numpy array.
  TypeError: The input action names should be two list.
  ```

  Will raise ValueError if shapes of inputs do not match :

  ```
  ValueError: The input two payoff matrices should have same size.
  ValueError: The length of input action names should match the number of actions.
  ```

## Output

```python
NESol = NESolver(A=A, B=B, action_name_1=n1, action_name_2=n2)
```

### NESolver.analyze()

```
NESol.analyze()
```

This method will print out the information of strictly dominated actions(SDA), pure Nash Equilibria(PNE) and mixed Nash Equilibria(MNE) of a given game.

Example output:

```
======= Analyze NE =======
n_SDA:	2
1_SDA:	['I']
2_SDA:	['X']
---------------
n_PNE:	2
PNE:
	(('J',), ('Y',))
	(('F',), ('Z',))
---------------
n_MNE:	1
MNE:
            support  NE_count                                                          NE_value
0  ((J, F), (Y, Z))         1  [(0.000000, 0.833333, 0.166667), (0.000000, 0.083333, 0.916667)]
```

* Strictly dominated actions

  * `n_SDA` :  The total number of strictly dominated actions of the two players.
  * `1_SDA` :  Strictly dominated actions of player 1. (action `I` in this example)
  * `2_SDA` :  Strictly dominated actions of player 2. (action `X` in this example)

* Pure Nash Equilibria

  * `n_PNE` :  Number of pure Nash Equilibria of the given game. (2 PNE in this example)
  * `PNE` :  All pure Nash Equilibria of the given game. ((J, Y) and (F, Z) are the only two PNE in this example)

* Mixed Nash Equilibria

  * `n_MNE` :  Number of mixed Nash Equilibria of the given game. (1 MNE in this example)

  * `MNE` :   All mixed Nash Equilibria of the given game.

    * Index column :  Start from 0.

    * `support` :  Actions with positive possibility. 

    * `NE_count` :  The number of Nash Equilibria based on the support :  `1` or `inf`.

    * `NE_value` :  If `NE_count==1`, it is the value of the MNE. If `NE_count==inf`, it shows an example of the MNE based on the support.

      ​						 

### NESolver.find()

```
info = NESol.find()
print(info)
```

This methods returns a Pandas Dataframe that contains detailed information on calculating the Nash Equilibrium based on each support.

Example output:

```
                   support NE_type  NE_count                                                          NE_value       w1        w2                         x_value  x_count                         y_value  y_count
0             ((I,), (X,))      -1         0                                                              None  8.00000       NaN                            None      0.0  (1.000000, 0.000000, 0.000000)      1.0
1             ((I,), (Y,))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
2             ((I,), (Z,))      -1         0                                                              None      NaN  4.000000  (1.000000, 0.000000, 0.000000)      1.0                            None      0.0
3           ((I,), (X, Y))      -1         0                                                              None  7.99994       NaN                            None      0.0  (0.999990, 0.000010, 0.000000)      inf
4           ((I,), (X, Z))      -1         0                                                              None  7.99994       NaN                            None      0.0  (0.999990, 0.000000, 0.000010)      inf
5           ((I,), (Y, Z))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
6        ((I,), (X, Y, Z))      -1         0                                                              None  7.99988       NaN                            None      0.0  (0.999980, 0.000010, 0.000010)      inf
7             ((J,), (X,))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
8             ((J,), (Y,))     PNE         1  [(0.000000, 1.000000, 0.000000), (0.000000, 1.000000, 0.000000)]  9.00000  5.000000  (0.000000, 1.000000, 0.000000)      1.0  (0.000000, 1.000000, 0.000000)      1.0
9             ((J,), (Z,))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
10          ((J,), (X, Y))      -1         0                                                              None  8.99994       NaN                            None      0.0  (0.000010, 0.999990, 0.000000)      inf
11          ((J,), (X, Z))      -1         0                                                              None  3.00000       NaN                            None      0.0  (0.166667, 0.000000, 0.833333)      inf
12          ((J,), (Y, Z))      -1         0                                                              None  8.99994       NaN                            None      0.0  (0.000000, 0.999990, 0.000010)      inf
13       ((J,), (X, Y, Z))      -1         0                                                              None  5.49997       NaN                            None      0.0  (0.583328, 0.416662, 0.000010)      inf
14            ((F,), (X,))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
15            ((F,), (Y,))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
16            ((F,), (Z,))     PNE         1  [(0.000000, 0.000000, 1.000000), (0.000000, 0.000000, 1.000000)]  4.00000  7.000000  (0.000000, 0.000000, 1.000000)      1.0  (0.000000, 0.000000, 1.000000)      1.0
17          ((F,), (X, Y))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
18          ((F,), (X, Z))      -1         0                                                              None  3.99994       NaN                            None      0.0  (0.000010, 0.000000, 0.999990)      inf
19          ((F,), (Y, Z))      -1         0                                                              None  3.99994       NaN                            None      0.0  (0.000000, 0.000010, 0.999990)      inf
20       ((F,), (X, Y, Z))      -1         0                                                              None  3.99988       NaN                            None      0.0  (0.000010, 0.000010, 0.999980)      inf
21          ((I, J), (X,))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
22          ((I, J), (Y,))      -1         0                                                              None      NaN  4.999970  (0.000010, 0.999990, 0.000000)      inf                            None      0.0
23          ((I, J), (Z,))      -1         0                                                              None      NaN  4.000000  (0.999990, 0.000010, 0.000000)      inf                            None      0.0
24        ((I, J), (X, Y))      -1         0                                                              None  5.50000       NaN                            None      0.0  (0.583333, 0.416667, 0.000000)      1.0
25        ((I, J), (X, Z))      -1         0                                                              None  3.00000       NaN                            None      0.0  (0.166667, 0.000000, 0.833333)      1.0
26        ((I, J), (Y, Z))      -1         0                                                              None      NaN  4.000000  (0.333333, 0.666667, 0.000000)      1.0                            None      0.0
27     ((I, J), (X, Y, Z))      -1         0                                                              None  5.49997       NaN                            None      0.0  (0.583328, 0.416662, 0.000010)      inf
28          ((I, F), (X,))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
29          ((I, F), (Y,))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
30          ((I, F), (Z,))      -1         0                                                              None      NaN  6.999970  (0.000010, 0.000000, 0.999990)      inf                            None      0.0
31        ((I, F), (X, Y))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
32        ((I, F), (X, Z))      -1         0                                                              None  3.00000       NaN                            None      0.0  (0.166667, 0.000000, 0.833333)      1.0
33        ((I, F), (Y, Z))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
34     ((I, F), (X, Y, Z))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
35          ((J, F), (X,))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
36          ((J, F), (Y,))      -1         0                                                              None      NaN  4.999970  (0.000000, 0.999990, 0.000010)      inf                            None      0.0
37          ((J, F), (Z,))      -1         0                                                              None      NaN  6.999970  (0.000000, 0.000010, 0.999990)      inf                            None      0.0
38        ((J, F), (X, Y))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
39        ((J, F), (X, Z))      -1         0                                                              None  3.00000       NaN                            None      0.0  (0.166667, 0.000000, 0.833333)      1.0
40        ((J, F), (Y, Z))     MNE         1  [(0.000000, 0.833333, 0.166667), (0.000000, 0.083333, 0.916667)]  3.50000  4.500000  (0.000000, 0.833333, 0.166667)      1.0  (0.000000, 0.083333, 0.916667)      1.0
41     ((J, F), (X, Y, Z))      -1         0                                                              None  3.49997       NaN                            None      0.0  (0.000010, 0.083328, 0.916662)      inf
42       ((I, J, F), (X,))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
43       ((I, J, F), (Y,))      -1         0                                                              None      NaN  4.999940  (0.000010, 0.999980, 0.000010)      inf                            None      0.0
44       ((I, J, F), (Z,))      -1         0                                                              None      NaN  4.000030  (0.999980, 0.000010, 0.000010)      inf                            None      0.0
45     ((I, J, F), (X, Y))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
46     ((I, J, F), (X, Z))      -1         0                                                              None  3.00000       NaN                            None      0.0  (0.166667, 0.000000, 0.833333)      1.0
47     ((I, J, F), (Y, Z))      -1         0                                                              None      NaN  4.499985  (0.000010, 0.833328, 0.166662)      inf                            None      0.0
48  ((I, J, F), (X, Y, Z))      -1         0                                                              None      NaN       NaN                            None      0.0                            None      0.0
```

* Meaning of the columns

  * Index column: Start from 0.

  * `support` :  Actions with positive possibility (done by support numeration).

  * `NE_type` :  The type of the Nash Equilibrium. 3 Possible values :

    * `-1`:  The support does not admit a Nash Equilibrium.
    * `PNE` :  The support admits an pure Nash Equilibrium.
    * `MNE` :   The support admits mixed Nash Equilibria.

  * `NE_count` :  The number of Nash Equilibria supported by the `support`. 3 Possible values :

    * `0` :  The support does not admit a Nash Equilibrium.
    * `1` :  The support admits one Nash Equilibrium.
    * `inf` :  The support admits infinitely many Nash Equilibria.

  * `NE_value` :  Value of the Nash Equilibria. 

    * If `NE_count==0`, it shows `None`.
    * If `NE_count==1`, it is the value of the MNE. 
    * If `NE_count==inf`, it shows an example of the MNE based on the support.

  * `w1` :  Expected payoff of **player 1**. `NaN` if **player 2** fails to have a strategy `y` to form a Nash Equilibrium.

    `w2` :  Expected payoff of **player 2**. `NaN` if **player 1** fails to have a strategy `x` to form a Nash Equilibrium.

  * `x_value` :  A stochastic vector representing (an example of) player1's strategy or `None` if player1 fails to have one.

    `y_value` :  A stochastic vector representing (an example of) player2's strategy or `None`  if player2 fails to have one.

  * `x_count`:  The number of player1's possible strategies :  `0.0`, `1.0` or `inf`

    `y_count`:  The number of player2's possible strategies :  `0.0`, `1.0` or `inf`

## Please see other examples in Test3.py

