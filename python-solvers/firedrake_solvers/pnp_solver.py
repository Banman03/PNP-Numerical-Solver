#!/usr/bin/env python3
# firedrake_equiv.py
# Converted from dolfinx -> firedrake
from firedrake import *
import numpy as np
from mpi4py import MPI

# Physical constants (same as original)
epsilon0  = 8.8541878128e-12
e0        = 1.60217663e-19
N_A       = 6.0221408e+23
kB        = 1.380649e-23
pot       = -1.2
T         = 2.98e2
pzc       = 0.0
potential = (pot - pzc) * e0 / kB / T
delta_RP  = 3.2e-10
epsilon_RP= 4.17
epsilon_s = 78.5

n0   = N_A
D0   = 1.0e-9
lambda_d = np.sqrt(epsilon_s * epsilon0 * kB * T / e0**2 / n0)
Ld   = lambda_d * 60.0
factor_a = delta_RP * epsilon_s / epsilon_RP / lambda_d
factor_1 = Ld / lambda_d
factor_2 = Ld * lambda_d / D0
ns_ions = 2
Length_limit = Ld / lambda_d

# Make mesh (square [0, Length_limit] x [0, Length_limit])
# Use a simple structured rectangle mesh
nx = ny = 32
mesh = RectangleMesh(nx, ny, Length_limit, Length_limit, quadrilateral=False)

# Mixed function space: three scalar CG1 fields
V0 = FunctionSpace(mesh, "CG", 1)
V = V0 * V0 * V0

# Spatial coordinate (ufl)
x = SpatialCoordinate(mesh)

# Define the analytic fields c1, c2, phi (ufl expressions)
c1 = 1.0 + 0.5 * sin(pi * x[0] / Length_limit) * sin(pi * x[1] / Length_limit)
c2 = 1.0 - 0.5 * sin(pi * x[0] / Length_limit) * sin(pi * x[1] / Length_limit)
phi = sin(pi * x[0] / Length_limit) * sin(pi * x[1] / Length_limit)

# diffusion coefficients from original code
d1 = 9.311000000000e+00
d2 = 5.273000000000e+00

# Precompute necessary UFL pieces for the source terms
dotc1 = inner(grad(c1), grad(phi))
dotc2 = inner(grad(c2), grad(phi))
s1 = factor_1 * d1 * (div(grad(c1)) + 1.0 * dotc1 + 1.0 * c1 * div(grad(phi)))
s2 = factor_1 * d2 * (div(grad(c2)) - 1.0 * dotc2 - 1.0 * c2 * div(grad(phi)))
p  = div(grad(phi)) - (1.0 * c1 - 1.0 * c2)

# Boundary indicator for right (x = Length_limit) and left (x = 0)
# Firedrake's DirichletBC allows a function that tests coordinates
eps = 1e-12
def right_boundary(xcoord):
    return np.isclose(xcoord[0], Length_limit, atol=1e-12)

def left_boundary(xcoord):
    return np.isclose(xcoord[0], 0.0, atol=1e-12)

# Create the mixed function and test/trial splits
u = Function(V, name="u")     # unknown (will hold (H, OH, phi))
v_H, v_OH, v_phi = TestFunctions(V)
u_H, u_OH, u_phi = split(u)

# Initial guess function (u_n in original)
u_n = Function(V, name="u_n")
# Interpolate initial guesses into subfunctions
# For mixed Function, use split assignment through intermediate Functions
u_n_H = Function(V0)
u_n_OH = Function(V0)
u_n_phi = Function(V0)

u_n_H.interpolate(Constant(1.0e-4))
u_n_OH.interpolate(Constant(1.0e-4))
# the original had linear expression for phi initial: 7.944060176975e-03*x + (-4.672608459440e+01)
# we map X[0] (x coordinate) appropriately (units consistent with domain)
x, y = SpatialCoordinate(u_n_phi.function_space().mesh())
u_n_phi.interpolate(7.944060176975e-03 * x + Constant(-4.672608459440e+01))

# Assign into the mixed u_n
assign(u_n.sub(0), u_n_H)
assign(u_n.sub(1), u_n_OH)
assign(u_n.sub(2), u_n_phi)

# Create Dirichlet BCs on the right boundary (x = Length_limit)
# Values: 1e-4 for H and OH, 0.0 for phi
bc_H  = DirichletBC(V.sub(0), Constant(1.0e-4), right_boundary)
bc_OH = DirichletBC(V.sub(1), Constant(1.0e-4), right_boundary)
bc_phi= DirichletBC(V.sub(2), Constant(0.0),     right_boundary)
bcs = [bc_H, bc_OH, bc_phi]

# Constants from original
flux_OH_at_left = Constant(1.0e-5)   # not used explicitly in this conversion (kept for completeness)
dt = Constant(1.0e-10)

# Define fluxes
flux_H  = d1 * (u_H * (+1.0) * grad(u_phi) + grad(u_H))
flux_OH = d2 * (u_OH * (-1.0) * grad(u_phi) + grad(u_OH))

# Variational forms (F = 0)
F_H   = factor_1 * dot(flux_H, grad(v_H)) * dx - s1 * v_H * dx
F_OH  = factor_1 * dot(flux_OH, grad(v_OH)) * dx - s2 * v_OH * dx
F_phi = dot(grad(u_phi), grad(v_phi)) * dx - ((+1.0) * u_H + (-1.0) * u_OH) * v_phi * dx - p * v_phi * dx

F = F_H + F_OH + F_phi

# Create nonlinear problem & solver (Firedrake style)
problem = NonlinearVariationalProblem(F, u, bcs=bcs)
# Solver parameters: use PETSc SNES with Newton by default
solver_parameters = {
    "snes_type": "newtonls",
    "snes_rtol": 1e-6,
    # linear solve options will be passed to KSP; choose defaults or override via PETSc options
    "ksp_type": "cg",
    "pc_type": "lu",  # you can change to "gamg" or others if you have parallel capabilities
}
solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

# Optionally expose PETSc options to be set from the command-line
# e.g., mpirun -n 4 python firedrake_equiv.py -snes_type newtonls -ksp_type gmres ...
# Solve
print("Before solve")
solver.solve()
print("After solve")

# Compute L2 differences between solution and analytic fields c1, c2, phi
# Extract components of the solution
uh, uoh, up = u.split()

H_L2  = sqrt(assemble(((uh - c1) ** 2) * dx))
OH_L2 = sqrt(assemble(((uoh - c2) ** 2) * dx))
phi_L2= sqrt(assemble(((up - phi) ** 2) * dx))

print(f"L2 errors: H={H_L2:.3e}  OH={OH_L2:.3e}  phi={phi_L2:.3e}")

# Save solution to VTK for inspection (Paraview)
File("firedrake_solution.pvd").write(u)
