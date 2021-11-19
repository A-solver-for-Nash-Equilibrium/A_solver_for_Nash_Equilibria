"""
## notes

lpm.update()
print(lpm.getVars())
lpm2 = lpm.copy()

z= np.array([lpm2.getVarByName("x[0]"),lpm2.getVarByName("x[1]")])  # cannot directly return whole vector x

try:
    lpm.optimize()
    for v in lpm.getVars():
        print('%s %g' % (v.varName, v.x))
except Exception as err:
    print("Oops. " + str(err) + ", this support does not admit a single Nash Equilibrium")
else:
    pass
finally:
    print("finish solving a LP")

print(float('inf'))
print(sys.float_info.max)
print(sys.float_info.epsilon)
print(sys.float_info.min)
"""

import sys
from gurobipy import GRB
import gurobipy as gp

sys.argv[0]
e = 0.000001 # 0.00001
e = float(sys.argv[1])


lpm = gp.Model("lpm")
lpm.params.NonConvex = 2

x1 = lpm.addVar(name='x1')
x2 = lpm.addVar(name='x2')
# y = lpm.addVar(name='y')
lpm.addConstr(x1 + x2 == 1)
# lpm.addConstr(x1 * y == 1)
lpm.addConstr(x1 == 1)
lpm.addConstr(x2 >= e) # x1 >= 0.00001
lpm.setObjective(0,GRB.MAXIMIZE)

lpm.optimize()
# print(lpm.getVarByName('y').x)
print("value of e: {e_}".format(e_=e))
print("value of x1: {x1_}".format(x1_=lpm.getVarByName('x1').x))
print("value of x2: {x2_}".format(x2_=lpm.getVarByName('x2').x))
