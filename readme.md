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

## Input

***class NESolver(A, B, action_name_1=None, action_name_2=None)***


* A :  2-d numpy array, payoff matirx of player 1

  B :  2-d numpy array, payoff matirx of player 2
  
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
from NESolver.NESolver4 import NESolver4 as NESolver
NESol = NESolver(A=A, B=B, action_name_1=n1, action_name_2=n2)
```

### NESolver.analyze()

```
NESol.analyze()
```

This is method will print out the information of strictly dominated actions(SDA), pure Nash Equilibria(PNE) and mixed Nash Equilibria(MNE) of this game.

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

