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

* A :  numpy array, payoff matirx of player 1

  B :  numpy array, payoff matirx of player 2

```python
A = np.array([[8, 2, 2], [3, 9, 3], [-2, -2, 4]])
B = np.array([[1, 2, 4], [-2, 5, 4], [-2, 2, 7]])
```

* a :  list (optional), action names of player1

  b :  list (optional), action names of player2

```python
a = ['I', 'J', 'F']
b = ['X', 'Y', 'Z']
```

