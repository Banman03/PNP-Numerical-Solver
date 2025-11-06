
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

# from firedrake.adjoint import *
# from firedrake import *
# from pyadjoint import minimize

# continue_annotation()

# n = 30
# mesh = UnitIntervalMesh(n)
# timestep = Constant(1.0/n)
# steps = 10

# x, = SpatialCoordinate(mesh)
# V = FunctionSpace(mesh, "CG", 2)
# ic = project(sin(2.*pi*x), V, name="ic")

# u_old = Function(V, name="u_old")
# u_new = Function(V, name="u")
# v = TestFunction(V)
# u_old.assign(ic)
# nu = Constant(0.0001)
# F = ((u_new-u_old)/timestep*v
#      + u_new*u_new.dx(0)*v + nu*u_new.dx(0)*v.dx(0))*dx
# bc = DirichletBC(V, 0.0, "on_boundary")
# problem = NonlinearVariationalProblem(F, u_new, bcs=bc)
# solver = NonlinearVariationalSolver(problem)

# # --- Known-solution setup for parameter inference ---

# # 1. Define the "true" diffusion coefficient (used to generate synthetic data)
# nu_true = Constant(0.0001)

# # 2. Generate the true solution (this is your synthetic "data")
# u_true = Function(V, name="u_true")
# u_old.assign(ic)
# for _ in range(steps):
#     F_true = ((u_true - u_old)/timestep*v
#               + u_true*u_true.dx(0)*v + nu_true*u_true.dx(0)*v.dx(0))*dx
#     problem_true = NonlinearVariationalProblem(F_true, u_true, bcs=bc)
#     solver_true = NonlinearVariationalSolver(problem_true)
#     solver_true.solve()
#     u_old.assign(u_true)
# u_data = u_true.copy(deepcopy=True)  # this is our "observed" solution

# # 3. Reset the initial condition and define a *guess* for nu
# u_old.assign(ic)
# nu_guess = Constant(0.001)  # wrong initial guess (to be optimized)
# nu_control = Control(nu_guess)

# # 4. Redefine PDE using the guessed nu
# F = ((u_new - u_old)/timestep*v
#      + u_new*u_new.dx(0)*v + nu_guess*u_new.dx(0)*v.dx(0))*dx
# problem = NonlinearVariationalProblem(F, u_new, bcs=bc)
# solver = NonlinearVariationalSolver(problem)

# # 5. Define the misfit functional (difference between model and data)
# J = 0
# for _ in range(steps):
#     solver.solve()
#     u_old.assign(u_new)
#     J += assemble(0.5*(u_new - u_data)**2*dx)

# # 6. Wrap the functional and control for adjoint-based optimization
# Jhat = ReducedFunctional(J, nu_control)

# # 7. Perform parameter inference (minimize J with respect to nu)
# nu_opt = minimize(Jhat)
# print("Optimized nu =", float(nu_opt))


from firedrake.adjoint import *
from firedrake import *
from pyadjoint import minimize, AdjFloat
from pyadjoint.optimization import optimization_problem, optimization_solver

continue_annotation()
n = 30
mesh = UnitIntervalMesh(n)
timestep = Constant(1.0/n)
steps = 10
x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 2)
ic = project(sin(2.*pi*x), V, name="ic")

# True diffusion coefficient
nu_true = Constant(0.0001)

# Generate synthetic data
u_true = Function(V, name="u_true")
u_old = Function(V, name="u_old")
u_old.assign(ic)
v = TestFunction(V)
bc = DirichletBC(V, 0.0, "on_boundary")

for _ in range(steps):
    F_true = ((u_true - u_old)/timestep*v
              + u_true*u_true.dx(0)*v + nu_true*u_true.dx(0)*v.dx(0))*dx
    problem_true = NonlinearVariationalProblem(F_true, u_true, bcs=bc)
    solver_true = NonlinearVariationalSolver(problem_true)
    solver_true.solve()
    u_old.assign(u_true)

u_data = u_true.copy(deepcopy=True)

# Reset and setup for parameter inference
u_old.assign(ic)
u_new = Function(V, name="u")

# Use a Function in Real space for the control parameter
R = FunctionSpace(mesh, "R", 0)
nu_guess = Function(R, name="nu")
nu_guess.assign(0.001)
nu_control = Control(nu_guess)

# Define misfit functional - build PDE with nu_guess directly
J = 0
for _ in range(steps):
    F = ((u_new - u_old)/timestep*v
         + u_new*u_new.dx(0)*v + nu_guess*u_new.dx(0)*v.dx(0))*dx
    problem = NonlinearVariationalProblem(F, u_new, bcs=bc)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()
    u_old.assign(u_new)
    J += assemble(0.5*(u_new - u_data)**2*dx)

pause_annotation()

# Optimize
Jhat = ReducedFunctional(J, nu_control)
nu_opt = minimize(Jhat)
print(f"True nu = {float(nu_true)}")
print(f"Optimized nu = {float(nu_opt)}")