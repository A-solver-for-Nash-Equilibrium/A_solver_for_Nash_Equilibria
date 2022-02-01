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

import numpy as np
a = list(range(5))
print(a)
print(type(a))
print(isinstance(a, list))





