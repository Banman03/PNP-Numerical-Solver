
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

from firedrake.adjoint import *
from firedrake import *

continue_annotation()

n = 30
mesh = UnitIntervalMesh(n)
timestep = Constant(1.0/n)
steps = 10

x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 2)
ic = project(sin(2.*pi*x), V, name="ic")

u_old = Function(V, name="u_old")
u_new = Function(V, name="u")
v = TestFunction(V)
u_old.assign(ic)
nu = Constant(0.0001)
F = ((u_new-u_old)/timestep*v
     + u_new*u_new.dx(0)*v + nu*u_new.dx(0)*v.dx(0))*dx
bc = DirichletBC(V, 0.0, "on_boundary")
problem = NonlinearVariationalProblem(F, u_new, bcs=bc)
solver = NonlinearVariationalSolver(problem)

J = assemble(ic*ic*dx)

for _ in range(steps):
    solver.solve()
    u_old.assign(u_new)
    J += assemble(u_new*u_new*dx)
pause_annotation()
print(round(J, 3))

Jhat = ReducedFunctional(J, Control(ic))

ic_new = project(sin(pi*x), V)
J_new = Jhat(ic_new)
print(round(J_new, 3))

get_working_tape().progress_bar = ProgressBar

dJ = Jhat.derivative()


# now let's perform some parameter inference :))))
