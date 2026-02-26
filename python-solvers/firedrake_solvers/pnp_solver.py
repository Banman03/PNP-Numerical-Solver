from firedrake import *
import argparse
import json
import math as m

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# print("available files: ", os.listdir())

try:
    from pnp_plotter import plot_solutions, create_animations, prepare_results_folder
    from pnp_utils import generate_solver_params
except ImportError:
    from firedrake_solvers.pnp_plotter import plot_solutions, create_animations, prepare_results_folder
    from firedrake_solvers.pnp_utils import generate_solver_params

results_dir = "firedrake_solvers/solver_results_temp"

prepare_results_folder(results_dir)

parser = argparse.ArgumentParser(
    description='PNP solver with optional Butler-Volmer boundary conditions',
    epilog='Examples:\n  python pnp_solver.py 0  # Run without BV (default case)\n  python pnp_solver.py 1  # Run with Butler-Volmer BCs',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('bc_mode', type=int, choices=[0, 1, 2],
                    help='Type of boundary conditions: 2=robin, 1=butler-volmer, 0=dirichlet')
args = parser.parse_args()

BC_MODE_DEFAULT = 0
BC_MODE_BV = 1
BC_MODE_ROBIN = 2
mode = args.bc_mode

use_butler_volmer = (mode == BC_MODE_BV)
use_robin = (mode == BC_MODE_ROBIN)
print(f"\n{'='*60}")
print(f"Butler-Volmer boundary conditions: {'ENABLED' if use_butler_volmer else 'DISABLED'}")
print(f"{'='*60}\n")

n_species = n = 4
order = 1
dt = 1e-9
t_end = 2e-6
num_steps = int(t_end / dt) + 1
output_interval = 50

F = 96485.3329
R = 8.314462618
T = 298.15
F_over_RT = F/(R*T)

# Example for a Lithium Sulfate system (Li+, SO4 2-, H+, OH-)
z_vals = [1, -2, 1, -1]
D_vals = [1.03e-9, 1.07e-9, 9.311e-9, 5.273e-9]

# effected solvated radius
a_rad = [0.382e-9, 0.733e-9, 0.28e-9, 0.3e-9]
a_vol = [(4/3)*m.pi*(a**3) for a in a_rad]
a_vals = [6.022e23*a for a in a_vol]

print("a vals: ",a_vals)

phi_applied = Constant(0.5)

# Parameters for unique BCs
if use_butler_volmer:
    j0 = Constant(0.1)
    alpha = Constant(0.5)
    n_electrons = Constant(1.0)
    phi_eq = Constant(0.0)
elif use_robin:
    # Robin Parameters: Flux J = kappa * (c - c_inf)
    # kappa: Mass transfer coefficient (m/s)
    # c_inf: Ambient/Bulk concentration
    kappa = Constant(1.0)
    c_inf = Constant(0.01)

L_scale = 50e-9
mesh = UnitSquareMesh(32, 32)
ds = Measure("ds", domain=mesh)

t_ref = (L_scale**2) / dt
dt_tilde = dt / t_ref
t_end_tilde = t_end / t_ref

c_ref = 0.1
phi_ref = R * T / F
print("phi ref: ", phi_ref)
D_ref = 1e-9

D_tilde = [D / D_ref for D in D_vals]
print("d tilde: ", D_tilde)

unit_conv = Constant(0.6022) 
a_tilde = [a * c_ref * unit_conv for a in a_vals]

eps_val = 1e-8
beta_val = (F * c_ref * (L_scale**2)) / (eps_val * phi_ref)
print("beta: ", beta_val)
beta = Constant(beta_val)

electrode_marker = 1

V_scalar = FunctionSpace(mesh, "CG", order)
mixed_spaces = [V_scalar for _ in range(n)] + [V_scalar]  # last is phi
W = MixedFunctionSpace(mixed_spaces)

U = Function(W)
U_prev = Function(W)

splitCur = split(U)
splitPrev = split(U_prev)

v_tests = TestFunctions(W)

ci = split(U)[:-1]
phi = split(U)[-1]
ci_prev = split(U_prev)[:-1]
phi_prev = split(U_prev)[-1]

v_list = v_tests[:-1]
w = v_tests[-1]

# mu = kT * ln(1 - sum_j a_j * c_j)
kBT = Constant(R*T)

sum_a_c = sum( Constant(a_tilde[i]) * ci[i] for i in range(n) )
mu_steric = ln(1 - sum_a_c)

F_res = 0

# Nernst-Planck Equation
for i in range(n):
    c = ci[i]
    c_old = ci_prev[i]
    v = v_list[i]
    D = Constant(D_tilde[i])
    z = Constant(z_vals[i])
    
    F_res += ( (c - c_old)/dt_tilde * v )*dx

    drift_potential = z * phi + mu_steric
    Jflux = D*(grad(c) + c * grad(drift_potential))
    F_res += dot(Jflux, grad(v))*dx

# Poisson Equation
phi_test = w
F_res += dot(grad(phi), grad(phi_test))*dx
F_res -= beta * sum( Constant(z_vals[i]) * ci[i] * phi_test for i in range(n) )*dx

# Unique BC boundary flux
# phi_applied_tilde = Constant(0.5 / phi_ref)
phi_applied_tilde = Constant(0.5)
print("phi applied: ", phi_applied_tilde)
J_ref = D_ref * c_ref / L_scale

if use_butler_volmer:
    j0_physical = 0.1
    j0_tilde_val = (j0_physical * L_scale) / (n_electrons.values()[0] * F * J_ref)
    j0_tilde = Constant(j0_tilde_val)
    
    # phi_applied_tilde is the metal potential. phi is the electrolyte potential.
    eta = phi_applied_tilde - phi - phi_eq
    
    j_BV_tilde = j0_tilde * ( exp(-alpha * n_electrons * eta) - exp((1 - alpha) * n_electrons * eta) )

    for i in range(n):
        v = v_list[i]
        if i == 0:
            F_res += ( j_BV_tilde * v ) * ds(electrode_marker)
        else:
            F_res -= ( j_BV_tilde * v ) * ds(electrode_marker)
            
elif use_robin:
    # Scale kappa: kappa_tilde = kappa * L_ref / D_ref
    kappa_physical = 1.0
    kappa_tilde = Constant(kappa_physical * L_scale / D_ref)
    
    # c_inf scaled by c_ref
    c_inf_tilde = Constant(0.01 / c_ref)
    
    for i in range(n):
        v = v_list[i]
        c = ci[i]
        F_res += kappa_tilde * (c - c_inf_tilde) * v * ds(electrode_marker)

c0_tilde = 1.0

if use_butler_volmer:
    bc_phi_ground = DirichletBC(W.sub(n), Constant(0.0), 2)
    # bc_ci = [DirichletBC(W.sub(i), Constant(c0_tilde), 2) for i in range(n)]
    bcs = [bc_phi_ground]
elif use_robin:
    bc_phi_electrode = DirichletBC(W.sub(n), phi_applied_tilde, 1)
    bc_phi_ground = DirichletBC(W.sub(n), Constant(0.0), 2)
    bc_ci = [DirichletBC(W.sub(i), Constant(c0_tilde), 3) for i in range(n)]
    bc_ci += [DirichletBC(W.sub(i), Constant(c0_tilde), 4) for i in range(n)]
    bcs = bc_ci + [bc_phi_electrode, bc_phi_ground]
else:
    # Dirichlet bcs
    bc_phi_electrode = DirichletBC(W.sub(n), Constant(0.01), 1)
    phi0 = 0.0
    # bc_phi = DirichletBC(W.sub(n), Constant(phi0), 1)
    bc_ci = [DirichletBC(W.sub(i), Constant(c0_tilde), 1) for i in range(n)]
    bc_ci += [DirichletBC(W.sub(i), Constant(c0_tilde), 2) for i in range(n)]
    bc_ci += [DirichletBC(W.sub(i), Constant(c0_tilde), 3) for i in range(n)]
    bc_ci += [DirichletBC(W.sub(i), Constant(c0_tilde), 4) for i in range(n)]

    bcs = bc_ci + [bc_phi_electrode]

J = derivative(F_res, U)

for i in range(n):
    U_prev.sub(i).assign(Constant(c0_tilde))

if use_butler_volmer or use_robin:
    x, y = SpatialCoordinate(mesh)
    phi_init = phi_applied_tilde * (1 - y)
    U_prev.sub(n).interpolate(phi_init)
    U.assign(U_prev)
else:
    U_prev.sub(n).assign(Constant(0.0))


# if use_butler_volmer or use_robin:
solver_param_array = generate_solver_params()
    
U_initial_state = Function(W).assign(U_prev)
    
for i, sp in enumerate(solver_param_array):
    if i is not 0: # I have added this for quick testing
        break
    
    U.assign(U_initial_state)
    U_prev.assign(U_initial_state)
    problem = NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
    
    print("solver params: ", json.dumps(sp, indent=2))
    # print(f"Testing Configuration {i}: {sp['snes_type']} + {sp['snes_linesearch_type']}")
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)

    t = 0.0
    print(f"Starting time evolution: {num_steps} steps from t=0 to t={t_end}")
    print(f"Time step dt = {dt}")

    snapshots = {'t': [], 'c0': [], 'c1': [], 'c2': [], 'c3': [], 'phi': []}

    if use_butler_volmer:
        suffix = "_bv"
    elif use_robin:
        suffix = "_robin"
    else:
        suffix = "_default"

    phi_file = VTKFile(f"phi{suffix}.pvd")
    c_files = [VTKFile(f"c{i}{suffix}.pvd") for i in range(n)]
    
    # will hold the values of the steric effects for inspection
    V_monitor = FunctionSpace(mesh, "CG", order)
    mu_monitor = Function(V_monitor, name="StericPotential")
    mu_file = VTKFile(f"mu_steric{suffix}.pvd")

    # print(f"\n{'*'*20} INITIAL CONDITIONS VERIFICATION {'*'*20}")

    # for i in range(n):
        # c_data = U_prev.sub(i).dat.data_ro
        # print(f"Species c{i}: Min={c_data.min():.4f}, Max={c_data.max():.4f}, Mean={c_data.mean():.4f}")

    # phi_data = U_prev.sub(n).dat.data_ro
    # print(f"Potential phi: Min={phi_data.min():.4f}, Max={phi_data.max():.4f}, Mean={phi_data.mean():.4f}")
    # print(f"{'*'*60}\n")

    # plot initial conditions if we want
    # plot_solutions(U_prev, z_vals, mode, num_steps, dt)

    for step in range(num_steps):
        try:
            solver.solve()
        except Exception as e:
            print(f"Failed to converge: {e}")
            break

        t += dt

        if step % output_interval == 0 or step == num_steps - 1:
            print(f"  Step {step+1}/{num_steps}, t = {t:.4f}")

            phi_file.write(U.sub(n), time=t)
            for i in range(n):
                c_files[i].write(U.sub(i), time=t)

            snapshots['t'].append(t)
            snapshots['c0'].append(U.sub(0).dat.data_ro.copy())
            snapshots['c1'].append(U.sub(1).dat.data_ro.copy())
            snapshots['c2'].append(U.sub(2).dat.data_ro.copy())
            snapshots['c3'].append(U.sub(3).dat.data_ro.copy())            
            snapshots['phi'].append(U.sub(n).dat.data_ro.copy())

            mu_monitor.project(mu_steric)
            mu_file.write(mu_monitor, time=t)
            mu_data = mu_monitor.dat.data_ro
            sum_a_c_val = assemble(sum_a_c * dx) / assemble(Constant(1.0) * dx(domain=mesh))
            
            print(f"  [Steric Check] Step {step+1}:")
            print(f"    - mu_steric: Min={mu_data.min():.10f}, Max={mu_data.max():.10f}, Mean={mu_data.mean():.10f}")
            print(f"    - Volumetric Occupancy (sum a*c): {sum_a_c_val:.10f}")

        U_prev.assign(U)

    print(f"Time evolution complete! Final time: t = {t:.4f}")

    plot_solutions(U_prev, z_vals, mode, num_steps, dt, t, results_dir, n)

    create_animations(snapshots, mode, mesh, results_dir)
    
    
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

Let's search for manufactured (numerical analysis) solution papers for the PNP equation. They will use a manufactured solution.
We want to be able to find the value of Si at each of the mesh's nodes.

The source term will go in the weak formulation, so we need to compute Si prior to implementing the manufactured solution.
'''



"""
If there is a strong potential gradient, that potential gradient is either supporting or inhibiting diffusion.
From the gif, the concentration seems to be diffusing upward, which seems like a strong break in symmetry.

"""