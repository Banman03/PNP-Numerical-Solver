from firedrake import *
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='PNP solver with optional Butler-Volmer boundary conditions',
    epilog='Examples:\n  python pnp_solver.py 0  # Run without BV (default case)\n  python pnp_solver.py 1  # Run with Butler-Volmer BCs',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('bv_enabled', type=int, choices=[0, 1],
                    help='Enable Butler-Volmer BCs: 1=on, 0=off')
args = parser.parse_args()

use_butler_volmer = bool(args.bv_enabled)
print(f"\n{'='*60}")
print(f"Butler-Volmer boundary conditions: {'ENABLED' if use_butler_volmer else 'DISABLED'}")
print(f"{'='*60}\n")

n_species = n = 2
order = 1
dt = 1e-3
t_end = 1.0  # end time
num_steps = int(t_end / dt)
output_interval = 50  # save output every N steps

F = 96485.3329
R = 8.314462618
T = 298.15
F_over_RT = F/(R*T)

D_vals = [1.0, 1.0]                # list length n
z_vals = [1, -1]
a_vals = [0.0, 0.0]                # steric sizes a_j; zero if no steric

# Butler-Volmer parameters (only used if use_butler_volmer=True)
if use_butler_volmer:
    j0 = Constant(0.01)             # exchange current density (reduced for stability)
    alpha = Constant(0.5)           # charge transfer coefficient
    n_electrons = Constant(1.0)     # electrons transferred
    phi_eq = Constant(0.0)          # equilibrium potential
    phi_applied = Constant(0.05)    # applied electrode potential

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

# Butler-Volmer boundary flux (only if enabled)
if use_butler_volmer:
    # For redox reaction: O (species 0, z=+1) + e⁻ ⇌ R (species 1, z=-1)
    # BV current density at electrode:
    eta = phi - phi_applied
    j_BV = j0*( exp(-alpha*n_electrons*F_over_RT*eta) - exp((1-alpha)*n_electrons*F_over_RT*eta) )

    # Flux boundary condition for each species
    for i in range(n):
        v = v_list[i]
        if i == 0:
            # Oxidized species: consumed at electrode (negative flux)
            F_res += ( (1.0/(n_electrons*F)) * j_BV * v )*ds(electrode_marker)
        else:
            # Reduced species: produced at electrode (positive flux)
            F_res -= ( (1.0/(n_electrons*F)) * j_BV * v )*ds(electrode_marker)

# Poisson residual
eps = Constant(1.0)  # permittivity (set appropriately)
phi_test = w
F_res += eps*dot(grad(phi), grad(phi_test))*dx
F_res -= sum( Constant(z_vals[i])*F * ci[i]*phi_test for i in range(n) )*dx
# add Neumann bc on phi if necessary: subtract flux * w over Gamma_N


c0 = 1.0

# Boundary conditions
if use_butler_volmer:
    # With BV: electrode at phi_applied, ground at opposite side
    bc_phi_electrode = DirichletBC(W.sub(n), phi_applied, 1)
    bc_phi_ground = DirichletBC(W.sub(n), Constant(0.0), 3)
    bc_ci = [DirichletBC(W.sub(i), Constant(c0), 3) for i in range(n)]
    bcs = bc_ci + [bc_phi_electrode, bc_phi_ground]
else:
    # Without BV: original boundary conditions
    phi0 = 0.0
    bc_phi = DirichletBC(W.sub(n), Constant(phi0), 1)
    bc_ci = [DirichletBC(W.sub(i), Constant(c0), 3) for i in range(n)]
    bcs = bc_ci + [bc_phi]

# --- form the Jacobian automatically
J = derivative(F_res, U)

# --- initial condition: fill U_prev with initial c's and phi
for i in range(n):
    U_prev.sub(i).assign(Constant(c0))

if use_butler_volmer:
    # Initialize phi with linear profile for better convergence
    x, y = SpatialCoordinate(mesh)
    phi_init = phi_applied * (1 - y)  # linear: phi_applied at y=0, 0 at y=1
    U_prev.sub(n).interpolate(phi_init)
    U.assign(U_prev)  # Initialize U as well
else:
    # Original: uniform phi=0
    U_prev.sub(n).assign(Constant(0.0))

# --- solve monolithically using NonlinearVariationalSolver
problem = NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)

if use_butler_volmer:
    # More robust solver parameters for BV nonlinearity
    solver_params = {
        'snes_type': 'newtonls',
        'snes_max_it': 100,
        'snes_rtol': 1e-6,
        'snes_atol': 1e-6,
        'snes_linesearch_type': 'bt',  # backtracking line search
        'snes_monitor': None,
        'ksp_type': 'gmres',
        'ksp_max_it': 200,
        'ksp_rtol': 1e-6,
        'pc_type': 'ilu',
    }
else:
    # Original solver parameters
    solver_params = {
        'snes_type': 'newtonls',
        'snes_max_it': 50,
        'snes_rtol': 1e-9,
        'ksp_type': 'preonly',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'additive',
    }

solver = NonlinearVariationalSolver(problem, solver_parameters=solver_params)

# --- Time-stepping loop
t = 0.0
print(f"Starting time evolution: {num_steps} steps from t=0 to t={t_end}")
print(f"Time step dt = {dt}")

# Storage for animation snapshots
snapshots = {'t': [], 'c0': [], 'c1': [], 'phi': []}

# Open VTK files for time series output
bv_suffix = "_bv" if use_butler_volmer else "_no_bv"
phi_file = VTKFile(f"phi{bv_suffix}.pvd")
c_files = [VTKFile(f"c{i}{bv_suffix}.pvd") for i in range(n)]

for step in range(num_steps):
    # Solve for current time step
    solver.solve()

    # Update time
    t += dt

    # Progress output
    if step % output_interval == 0 or step == num_steps - 1:
        print(f"  Step {step+1}/{num_steps}, t = {t:.4f}")

        # Write VTK output at intervals
        phi_file.write(U.sub(n), time=t)
        for i in range(n):
            c_files[i].write(U.sub(i), time=t)

        # Store snapshots for animation
        snapshots['t'].append(t)
        # Copy function data at vertices for plotting
        snapshots['c0'].append(U.sub(0).dat.data_ro.copy())
        snapshots['c1'].append(U.sub(1).dat.data_ro.copy())
        snapshots['phi'].append(U.sub(n).dat.data_ro.copy())

    # Copy current solution to previous for next time step
    U_prev.assign(U)

print(f"Time evolution complete! Final time: t = {t:.4f}")

# Extract actual Function objects for final visualization
c0_func = U.sub(0)
c1_func = U.sub(1)
phi_func = U.sub(2)

# --- Visualization with matplotlib
import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, tricontour

# Create figure with subplots for each field
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot concentration of species 0
tripcolor(c0_func, axes=axes[0], cmap='viridis')
axes[0].set_title(f'Concentration c₀ (z={z_vals[0]:+d}) at t={t:.3f}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_aspect('equal')
plt.colorbar(axes[0].collections[0], ax=axes[0], label='c₀')

# Plot concentration of species 1
tripcolor(c1_func, axes=axes[1], cmap='plasma')
axes[1].set_title(f'Concentration c₁ (z={z_vals[1]:+d}) at t={t:.3f}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_aspect('equal')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='c₁')

# Plot electric potential
tripcolor(phi_func, axes=axes[2], cmap='coolwarm')
axes[2].set_title(f'Electric Potential phi at t={t:.3f}')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_aspect('equal')
plt.colorbar(axes[2].collections[0], ax=axes[2], label='phi')

bv_status = "with BV" if use_butler_volmer else "no BV"
fig.suptitle(f'PNP Solution ({bv_status}): {num_steps} time steps, dt={dt}', fontsize=14, y=1.02)
plt.tight_layout()
output_png = f'pnp_solution{bv_suffix}.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Saved final state visualization to {output_png}")
plt.show()

# --- Create animations
print(f"\nGenerating animations from {len(snapshots['t'])} snapshots...")
import matplotlib.animation as animation
import numpy as np

# Get mesh coordinates for plotting
coords = mesh.coordinates.dat.data_ro
x_coords = coords[:, 0]
y_coords = coords[:, 1]

# Create triangulation for plotting
import matplotlib.tri as tri
triangulation = tri.Triangulation(x_coords, y_coords)

def create_animation(data_list, title, cmap, filename):
    """Create animation for a single field"""
    fig_anim, ax = plt.subplots(figsize=(6, 5))

    # Determine global colorbar limits
    vmin = min(np.min(d) for d in data_list)
    vmax = max(np.max(d) for d in data_list)

    # Initial plot
    tpc = ax.tripcolor(triangulation, data_list[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.colorbar(tpc, ax=ax, label=title)

    def update(frame):
        ax.clear()
        tpc = ax.tripcolor(triangulation, data_list[frame], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title(title)
        time_text = ax.text(0.02, 0.95, f't = {snapshots["t"][frame]:.3f}',
                           transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        return tpc, time_text

    anim = animation.FuncAnimation(fig_anim, update, frames=len(data_list),
                                   interval=100, blit=False, repeat=True)

    # Try to save as MP4, fall back to GIF if ffmpeg not available
    try:
        anim.save(filename + '.mp4', writer='ffmpeg', fps=10, dpi=150)
        print(f"  Saved {filename}.mp4")
    except:
        try:
            anim.save(filename + '.gif', writer='pillow', fps=10)
            print(f"  Saved {filename}.gif (ffmpeg not available)")
        except Exception as e:
            print(f"  Warning: Could not save animation for {title}: {e}")

    plt.close(fig_anim)

# Create animations for each field
create_animation(snapshots['c0'], 'Concentration c₀', 'viridis', f'c0_animation{bv_suffix}')
create_animation(snapshots['c1'], 'Concentration c₁', 'plasma', f'c1_animation{bv_suffix}')
create_animation(snapshots['phi'], 'Electric Potential phi', 'coolwarm', f'phi_animation{bv_suffix}')

print("Animation generation complete!")
    
    
    
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