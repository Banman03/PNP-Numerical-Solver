from firedrake import *

n_species = n = 2
order = 1
dt = 1e-3
F = 96485.3329
R = 8.314462618
T = 298.15
F_over_RT = F/(R*T)

D_vals = [1.0, 1.0]                # list length n
z_vals = [1, -1]
a_vals = [0.0, 0.0]                # steric sizes a_j; zero if no steric
j0 = Constant(1.0)                 # example for Butler-Volmer
alpha = Constant(0.5)
n_electrons = Constant(1.0)
phi_eq = Constant(0.0)

# --- mesh and measures
mesh = UnitSquareMesh(32, 32)
ds = Measure("ds", domain=mesh)

# assume electrode is boundary mark 1: use ds(1) in boundary integral
electrode_marker = 1

# --- function spaces: mixed for c_1,...,c_n, phi
V_scalar = FunctionSpace(mesh, "CG", order)
mixed_spaces = [V_scalar for _ in range(n)] + [V_scalar]  # last is phi
W = MixedFunctionSpace(mixed_spaces)

# --- trial / test / function
U = Function(W)
U_prev = Function(W)

splitCur = split(U)
splitPrev = split(U_prev)

v_tests = TestFunctions(W)
# Break out components
ci = split(U)[:-1]
phi = split(U)[-1]
ci_prev = split(U_prev)[:-1]
phi_prev = split(U_prev)[-1]

v_list = v_tests[:-1]
w = v_tests[-1]

# --- helper: mu_steric and its gradient
# mu = kT * ln(1 - sum_j a_j * c_j)
kB = Constant(1.380649e-23)  # if you really want k_B; often k_BT used -- adapt units
kBT = Constant(R*T)          # using R*T is common here (paper uses k_B T or R T depending)
sum_a_c = sum( Constant(a_vals[i]) * ci[i] for i in range(n) )
# mu_steric = kBT * ln(1 - sum_a_c)   # if a_vals are zero, this term vanishes
# grad(mu_steric) computed by UFL via grad(mu_steric)

# --- variational residual
F_res = 0
for i in range(n):
    c = ci[i]
    c_old = ci_prev[i]
    v = v_list[i]
    D = Constant(D_vals[i])
    z = Constant(z_vals[i])
    # time derivative (Backward Euler)
    F_res += ( (c - c_old)/dt * v )*dx

    # diffusion + drift (steric included)
    drift_potential = F_over_RT * z * phi # + mu_steric
    Jflux = D*(grad(c) + c * grad(drift_potential))
    F_res += dot(Jflux, grad(v))*dx

    # Butler-Volmer boundary flux (boundary integral on electrode)
    # BV current density:
    eta = phi - phi_eq
    j = j0*( exp(-alpha*n_electrons*F_over_RT*eta) - exp((1-alpha)*n_electrons*F_over_RT*eta) )
    # sign: choose + or - per reaction; here we add the contribution (paper uses +/-)
    # F_res -= ( (+1.0/(n_electrons*F)) * j * v )*ds(electrode_marker)

# Poisson residual
eps = Constant(1.0)  # permittivity (set appropriately)
phi_test = w
F_res += eps*dot(grad(phi), grad(phi_test))*dx
F_res -= sum( Constant(z_vals[i])*F * ci[i]*phi_test for i in range(n) )*dx
# add Neumann bc on phi if necessary: subtract flux * w over Gamma_N


phi0 = 0.0
c0 = 1.0

bc_phi = DirichletBC(W.sub(n), Constant(phi0), 1)
bc_ci  = [DirichletBC(W.sub(i), Constant(c0), 3) for i in range(n)]

bcs = bc_ci + [bc_phi]

# --- form the Jacobian automatically
J = derivative(F_res, U)

# --- initial condition: fill U_prev with initial c's and phi (e.g. uniform)
for i in range(n):
    U_prev.sub(i).assign(Constant(c0))
U_prev.sub(n).assign(Constant(phi0))

# --- solve monolithically using NonlinearVariationalSolver
problem = NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
solver_params = {
    'snes_type': 'newtonls',
    'snes_max_it': 50,
    'snes_rtol': 1e-9,
    'ksp_type': 'preonly',
    # use field split (block) preconditioner: separate species block(s) and potential block
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'additive',
    # setup fieldsplit names / splits afterwards in options context if needed
}

solver = NonlinearVariationalSolver(problem, solver_parameters=solver_params)
solver.solve()

components = split(U)
# concentrations
ci = components[:-1]
# potential
phi = components[-1]

phi_file = VTKFile("phi.pvd")
phi_file.write(phi)

for i, c in enumerate(ci):
    file = VTKFile(f"c{i}.pvd")
    file.write(c)
    
    
    
'''
Plot something as a magnitude, and, because it's a vector field, we can plot arrows going in different directions
You can also plot the potential on top of the concentrations

We can also look at different snapshots of the solution to see if its converging or not
We can also look at a time-dependent problem if the above doesn't work

To check convergence: Use L2 norm of true and FEM solution.
We can also use the H1 (tracks the error in the first derivative of the actual solution and FEM solution)
We can also use L2 trace and H1 trace (what is the error happening at the boundary) errors.
Refinement of the mesh (see if the error decreases as the grid is refined).
Sometimes if you use quadratic polynomials instead of piecewise linear can improve convergence.
'''