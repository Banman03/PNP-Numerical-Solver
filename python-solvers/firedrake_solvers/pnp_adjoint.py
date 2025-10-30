from firedrake import *
from firedrake.adjoint import *

continue_annotation()

# Info on the adjoint method in Firedrake: https://www.firedrakeproject.org/adjoint.html#equation-eq-djdm

# A functional evaluation of the underlying problem needs to first be computed.
#   This is done via a process called taping, in which the problem is solved, and pyadjoint records all subsequent operations 
#   The functional can be a variety of things. In the Firedrake example, It is the sum of the squared L2 norm at every timestep.
# 
# The reduced functional is the key to the adjoint method. It ties together a functional value (the result of a taped operation)
# and >= 1 controls which are an input of the computation of the functional value.


# Step by step:
"""
1. Define ICs, a mesh, function spaces, and trial and test functions, as you normally would when solving the PDE.
2. Perform a forward solve of the PDE.
3. Define a J (sum of squared errors is often used)
4. Define a target, typically the initial condition of the mesh. | ic = project(sin(2.*pi*x), V, name="ic"); Control(ic)
5. Create the reduced function form | jhat = ReducedFunctional(functional, i_c)
6. If desired compute the derivate of the functional with respect to the controls | djdm = jhat.derivative()
"""