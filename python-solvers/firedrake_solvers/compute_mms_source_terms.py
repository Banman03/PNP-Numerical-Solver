"""
Compute source terms for Method of Manufactured Solutions (MMS) for PNP equations.

Following MOOSE framework approach:
1. Define manufactured (exact) solutions
2. Substitute into strong form of PDEs
3. Compute forcing/source terms symbolically
4. Generate code for use in Firedrake solver

Reference: https://mooseframework.inl.gov/python/mms.html
"""

import sympy as sp
from sympy import symbols, sin, cos, exp, pi, diff, simplify, sqrt
import numpy as np

# Define symbolic variables
x, y, t = symbols('x y t', real=True)

# Physical parameters (matches pnp_solver.py)
F = 96485.3329
R = 8.314462618
T = 298.15
F_over_RT = F / (R * T)
eps = 1.0

# Diffusion coefficients and charges for n=2 species
D = [1.0, 1.0]
z = [1, -1]
n_species = 2

print("="*70)
print("Method of Manufactured Solutions - PNP Equations")
print("="*70)
print("\nPNP System:")
print("  Species equations: ∂c_i/∂t = ∇·[D_i(∇c_i + c_i z_i (F/RT) ∇φ)]")
print("  Poisson equation: -ε∇²φ = F Σ_i z_i c_i")
print("="*70)

# ============================================================================
# STEP 1: Define manufactured solutions
# ============================================================================
print("\n[1] Defining manufactured solutions...")
print("    (For spatial convergence: use trig/exponential functions)")

# Manufactured solution for concentrations (must be positive!)
# Using smooth functions that can have Dirichlet BCs applied
c_manufactured = []
for i in range(n_species):
    c_i = 1.0 + 0.1 * sin(pi * x) * sin(pi * y) * exp(-t)
    c_manufactured.append(c_i)
    print(f"  c_{i}(x,y,t) = {c_i}")

# Manufactured solution for electric potential
# Example: phi(x,y,t) = 0.1 * sin(pi*x) * sin(pi*y) * exp(-t)
phi_manufactured = 0.1 * sin(pi * x) * sin(pi * y) * exp(-t)
print(f"  phi(x,y,t) = {phi_manufactured}")

# ============================================================================
# STEP 2: Compute source terms from strong form
# ============================================================================
print("\n[2] Computing source/forcing terms from strong form...")
print("    (This makes the manufactured solution exact for the modified PDE)")

# For each species, the PNP equation strong form is:
# ∂c_i/∂t - ∇·[D_i (∇c_i + c_i z_i F/RT ∇φ)] = 0
#
# To use MMS, we add a forcing term S_i:
# ∂c_i/∂t - ∇·[D_i (∇c_i + c_i z_i F/RT ∇φ)] = S_i
#
# where S_i is computed by substituting manufactured solution

source_c = []
print("\n  Species source terms:")
for i in range(n_species):
    c_i = c_manufactured[i]
    D_i = D[i]
    z_i = z[i]

    print(f"\n  Species {i} (z={z_i:+d}, D={D_i}):")

    # Time derivative
    dc_dt = diff(c_i, t)

    # Gradient of c_i
    grad_c_x = diff(c_i, x)
    grad_c_y = diff(c_i, y)

    # Gradient of phi
    grad_phi_x = diff(phi_manufactured, x)
    grad_phi_y = diff(phi_manufactured, y)

    # Drift term coefficient
    drift_coeff = z_i * F_over_RT

    # Flux: J_i = D_i (∇c_i + c_i z_i F/RT ∇φ)
    J_x = D_i * (grad_c_x + c_i * drift_coeff * grad_phi_x)
    J_y = D_i * (grad_c_y + c_i * drift_coeff * grad_phi_y)

    # Divergence of flux
    div_J = diff(J_x, x) + diff(J_y, y)

    # Source term: S_i = ∂c_i/∂t - ∇·J_i
    # (Note: this is the RHS when strong form is written as: ∂c/∂t - ∇·J = S)
    S_i = dc_dt - div_J
    S_i_simplified = simplify(S_i)

    source_c.append(S_i_simplified)
    print(f"    S_{i} = {S_i_simplified}")

# For Poisson equation strong form:
# -∇·(ε ∇φ) - F Σ_i z_i c_i = 0
#
# With MMS forcing term:
# -∇·(ε ∇φ) - F Σ_i z_i c_i = S_phi

print("\n  Poisson source term:")

# Laplacian of phi: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y²
d2phi_dx2 = diff(phi_manufactured, x, 2)
d2phi_dy2 = diff(phi_manufactured, y, 2)
laplacian_phi = d2phi_dx2 + d2phi_dy2

# Charge density: ρ = F Σ_i z_i c_i
rho = sum(z[i] * c_manufactured[i] for i in range(n_species))

# Source term for Poisson: S_phi = -ε∇²φ - F Σ_i z_i c_i
source_phi = -eps * laplacian_phi - F * rho
source_phi_simplified = simplify(source_phi)

print(f"    S_phi = {source_phi_simplified}")

# ============================================================================
# STEP 3: Generate Firedrake-compatible code
# ============================================================================
print("\n[3] Generating Firedrake-compatible expressions...")

with open('mms_expressions.py', 'w') as f:
    f.write('"""\n')
    f.write('Generated MMS expressions for PNP solver\n')
    f.write('Use these in pnp_solver_mms.py for verification\n')
    f.write('\n')
    f.write('Method of Manufactured Solutions:\n')
    f.write('- Manufactured solutions are defined\n')
    f.write('- Source terms computed by substituting into strong form\n')
    f.write('- Add source terms to weak formulation\n')
    f.write('- Compare FEM solution to exact manufactured solution\n')
    f.write('"""\n\n')
    f.write('from firedrake import *\n')
    f.write('import numpy as np\n\n')

    # Physical parameters
    f.write('# Physical parameters\n')
    f.write(f'F = {F}\n')
    f.write(f'R = {R}\n')
    f.write(f'T = {T}\n')
    f.write(f'F_over_RT = {F_over_RT}\n')
    f.write(f'eps = {eps}\n')
    f.write(f'D_vals = {D}\n')
    f.write(f'z_vals = {z}\n')
    f.write(f'n_species = {n_species}\n\n')

    # Exact solutions as UFL expressions
    f.write('def get_exact_solutions(mesh, t_val):\n')
    f.write('    """\n')
    f.write('    Return exact (manufactured) solutions as Firedrake expressions.\n')
    f.write('    \n')
    f.write('    Args:\n')
    f.write('        mesh: Firedrake mesh\n')
    f.write('        t_val: current time value\n')
    f.write('    \n')
    f.write('    Returns:\n')
    f.write('        (c_exact_list, phi_exact): tuple of lists and expression\n')
    f.write('    """\n')
    f.write('    x, y = SpatialCoordinate(mesh)\n')
    f.write('    \n')
    f.write('    c_exact = []\n')
    for i in range(n_species):
        # Convert sympy expression to UFL-compatible string, replace 't' with 't_val'
        expr_str = str(c_manufactured[i])
        expr_str = expr_str.replace('t', 't_val')
        f.write(f'    c_exact.append({expr_str})  # c_{i}\n')
    f.write('    \n')
    expr_str = str(phi_manufactured).replace('t', 't_val')
    f.write(f'    phi_exact = {expr_str}\n')
    f.write('    \n')
    f.write('    return c_exact, phi_exact\n\n')

    # Source terms as UFL expressions
    f.write('def get_source_terms(mesh, t_val):\n')
    f.write('    """\n')
    f.write('    Return source/forcing terms as Firedrake expressions.\n')
    f.write('    These are added to the RHS of the weak formulation.\n')
    f.write('    \n')
    f.write('    Args:\n')
    f.write('        mesh: Firedrake mesh\n')
    f.write('        t_val: current time value\n')
    f.write('    \n')
    f.write('    Returns:\n')
    f.write('        (source_c_list, source_phi): tuple of lists and expression\n')
    f.write('    """\n')
    f.write('    x, y = SpatialCoordinate(mesh)\n')
    f.write('    \n')
    f.write('    source_c = []\n')
    for i in range(n_species):
        expr_str = str(source_c[i]).replace('t', 't_val')
        f.write(f'    source_c.append({expr_str})  # S_{i}\n')
    f.write('    \n')
    expr_str = str(source_phi).replace('t', 't_val')
    f.write(f'    source_phi = {expr_str}\n')
    f.write('    \n')
    f.write('    return source_c, source_phi\n\n')

    # Helper functions for numpy evaluation at nodes
    f.write('# ============================================================================\n')
    f.write('# Numpy functions for evaluation at mesh nodes\n')
    f.write('# ============================================================================\n\n')

    for i in range(n_species):
        f.write(f'def c{i}_exact_numpy(x, y, t):\n')
        f.write(f'    """Exact solution for c_{i}"""\n')
        expr_str = str(c_manufactured[i])
        expr_np = expr_str.replace('sin', 'np.sin').replace('cos', 'np.cos')
        expr_np = expr_np.replace('exp', 'np.exp').replace('pi', 'np.pi')
        f.write(f'    return {expr_np}\n\n')

    f.write(f'def phi_exact_numpy(x, y, t):\n')
    f.write(f'    """Exact solution for phi"""\n')
    expr_str = str(phi_manufactured)
    expr_np = expr_str.replace('sin', 'np.sin').replace('cos', 'np.cos')
    expr_np = expr_np.replace('exp', 'np.exp').replace('pi', 'np.pi')
    f.write(f'    return {expr_np}\n\n')

    for i in range(n_species):
        f.write(f'def source_c{i}_numpy(x, y, t):\n')
        f.write(f'    """Source term for c_{i}"""\n')
        expr_str = str(source_c[i])
        expr_np = expr_str.replace('sin', 'np.sin').replace('cos', 'np.cos')
        expr_np = expr_np.replace('exp', 'np.exp').replace('pi', 'np.pi')
        f.write(f'    return {expr_np}\n\n')

    f.write(f'def source_phi_numpy(x, y, t):\n')
    f.write(f'    """Source term for phi"""\n')
    expr_str = str(source_phi)
    expr_np = expr_str.replace('sin', 'np.sin').replace('cos', 'np.cos')
    expr_np = expr_np.replace('exp', 'np.exp').replace('pi', 'np.pi')
    f.write(f'    return {expr_np}\n')

print("  ✓ Saved to: mms_expressions.py")

# ============================================================================
# STEP 4: Test evaluation
# ============================================================================
print("\n[4] Testing source term evaluation...")

# Create lambdified functions
c_exact_funcs = []
for i in range(n_species):
    func = sp.lambdify((x, y, t), c_manufactured[i], 'numpy')
    c_exact_funcs.append(func)

phi_exact_func = sp.lambdify((x, y, t), phi_manufactured, 'numpy')

source_c_funcs = []
for i in range(n_species):
    func = sp.lambdify((x, y, t), source_c[i], 'numpy')
    source_c_funcs.append(func)

source_phi_func = sp.lambdify((x, y, t), source_phi, 'numpy')

# Test at specific point
x_test, y_test, t_test = 0.5, 0.5, 0.0
print(f"\n  At (x={x_test}, y={y_test}, t={t_test}):")

for i in range(n_species):
    c_val = c_exact_funcs[i](x_test, y_test, t_test)
    s_val = source_c_funcs[i](x_test, y_test, t_test)
    print(f"    c_{i}_exact = {c_val:.6e},  S_{i} = {s_val:.6e}")

phi_val = phi_exact_func(x_test, y_test, t_test)
s_phi_val = source_phi_func(x_test, y_test, t_test)
print(f"    phi_exact  = {phi_val:.6e},  S_phi = {s_phi_val:.6e}")

print("\n" + "="*70)
print("✓ MMS source term computation complete!")
print("\nNext steps:")
print("  1. Use mms_expressions.py in your Firedrake solver")
print("  2. Add source terms to weak formulation")
print("  3. Apply exact solutions as Dirichlet BCs")
print("  4. Compute L2/H1 error norms after solving")
print("  5. Perform mesh refinement study")
print("="*70)
