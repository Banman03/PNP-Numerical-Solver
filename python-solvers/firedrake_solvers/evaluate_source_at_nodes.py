"""
Evaluate MMS source terms at mesh nodes.

This script demonstrates how to:
1. Create a mesh
2. Evaluate source terms (forcing functions) at each node
3. Interpolate onto finite element functions
4. Visualize the source term distributions
"""

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from mms_expressions import get_source_terms, get_exact_solutions

# Create mesh
nx, ny = 16, 16
mesh = UnitSquareMesh(nx, ny)

# Create function space
order = 1
V = FunctionSpace(mesh, "CG", order)

# Time at which to evaluate
t_val = 0.0

print("="*70)
print("Evaluating MMS Source Terms at Mesh Nodes")
print("="*70)
print(f"Mesh: {nx} x {ny}")
print(f"Polynomial order: {order}")
print(f"Time: t = {t_val}")
print(f"Number of nodes: {mesh.num_vertices()}")
print(f"Number of DOFs: {V.dim()}")
print("="*70)

# Get source terms as UFL expressions
source_c_list, source_phi_expr = get_source_terms(mesh, t_val)

# Get exact solutions as well
c_exact_list, phi_exact_expr = get_exact_solutions(mesh, t_val)

# Create Functions to hold the interpolated values
S_c0 = Function(V, name="Source_c0")
S_c1 = Function(V, name="Source_c1")
S_phi = Function(V, name="Source_phi")

c0_exact = Function(V, name="c0_exact")
c1_exact = Function(V, name="c1_exact")
phi_exact = Function(V, name="phi_exact")

# Interpolate source terms onto the mesh
print("\nInterpolating source terms onto mesh...")
S_c0.interpolate(source_c_list[0])
S_c1.interpolate(source_c_list[1])
S_phi.interpolate(source_phi_expr)

# Interpolate exact solutions as well
c0_exact.interpolate(c_exact_list[0])
c1_exact.interpolate(c_exact_list[1])
phi_exact.interpolate(phi_exact_expr)

print("✓ Interpolation complete")

# Extract nodal values
S_c0_vals = S_c0.dat.data_ro
S_c1_vals = S_c1.dat.data_ro
S_phi_vals = S_phi.dat.data_ro

c0_vals = c0_exact.dat.data_ro
c1_vals = c1_exact.dat.data_ro
phi_vals = phi_exact.dat.data_ro

# Get mesh coordinates
coords = mesh.coordinates.dat.data_ro
x_coords = coords[:, 0]
y_coords = coords[:, 1]

print("\n" + "="*70)
print("Source Term Statistics")
print("="*70)
print(f"\nS_c0 (source for species 0):")
print(f"  min = {S_c0_vals.min():.6e}")
print(f"  max = {S_c0_vals.max():.6e}")
print(f"  mean = {S_c0_vals.mean():.6e}")
print(f"  std = {S_c0_vals.std():.6e}")

print(f"\nS_c1 (source for species 1):")
print(f"  min = {S_c1_vals.min():.6e}")
print(f"  max = {S_c1_vals.max():.6e}")
print(f"  mean = {S_c1_vals.mean():.6e}")
print(f"  std = {S_c1_vals.std():.6e}")

print(f"\nS_phi (source for Poisson):")
print(f"  min = {S_phi_vals.min():.6e}")
print(f"  max = {S_phi_vals.max():.6e}")
print(f"  mean = {S_phi_vals.mean():.6e}")
print(f"  std = {S_phi_vals.std():.6e}")

# Sample values at specific nodes
print("\n" + "="*70)
print("Sample Values at Specific Nodes")
print("="*70)

sample_indices = [0, len(x_coords)//4, len(x_coords)//2, 3*len(x_coords)//4, -1]
print(f"\n{'Node':>6} {'x':>10} {'y':>10} {'S_c0':>14} {'S_c1':>14} {'S_phi':>14}")
print("-"*70)
for idx in sample_indices:
    print(f"{idx:6d} {x_coords[idx]:10.4f} {y_coords[idx]:10.4f} "
          f"{S_c0_vals[idx]:14.6e} {S_c1_vals[idx]:14.6e} {S_phi_vals[idx]:14.6e}")

# Visualize source terms
print("\n" + "="*70)
print("Creating Visualizations")
print("="*70)

from firedrake.pyplot import tripcolor
import matplotlib.tri as tri

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Source terms
tripcolor(S_c0, axes=axes[0, 0], cmap='RdBu_r')
axes[0, 0].set_title('Source Term $S_{c_0}$', fontsize=14)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')

tripcolor(S_c1, axes=axes[0, 1], cmap='RdBu_r')
axes[0, 1].set_title('Source Term $S_{c_1}$', fontsize=14)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')

tripcolor(S_phi, axes=axes[0, 2], cmap='RdBu_r')
axes[0, 2].set_title('Source Term $S_{\phi}$', fontsize=14)
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('y')

# Row 2: Exact solutions
tripcolor(c0_exact, axes=axes[1, 0], cmap='viridis')
axes[1, 0].set_title('Exact Solution $c_0$', fontsize=14)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')

tripcolor(c1_exact, axes=axes[1, 1], cmap='viridis')
axes[1, 1].set_title('Exact Solution $c_1$', fontsize=14)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')

tripcolor(phi_exact, axes=axes[1, 2], cmap='coolwarm')
axes[1, 2].set_title('Exact Solution $\phi$', fontsize=14)
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('y')

plt.suptitle(f'MMS Source Terms and Exact Solutions (t={t_val})', fontsize=16)
plt.tight_layout()

output_file = 'mms_source_terms_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {output_file}")

# Save to VTK for ParaView
print("\nSaving to VTK for ParaView...")
vtk_file = VTKFile("mms_source_terms.pvd")
vtk_file.write(S_c0, S_c1, S_phi, c0_exact, c1_exact, phi_exact)
print(f"✓ Saved to: mms_source_terms.pvd")

# Export nodal values to CSV
print("\nExporting nodal values to CSV...")
import csv
csv_file = 'mms_source_terms_nodes.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['node_id', 'x', 'y', 'S_c0', 'S_c1', 'S_phi',
                     'c0_exact', 'c1_exact', 'phi_exact'])
    for i in range(len(x_coords)):
        writer.writerow([i, x_coords[i], y_coords[i],
                        S_c0_vals[i], S_c1_vals[i], S_phi_vals[i],
                        c0_vals[i], c1_vals[i], phi_vals[i]])

print(f"✓ Saved nodal values to: {csv_file}")
print(f"  Total nodes exported: {len(x_coords)}")

print("\n" + "="*70)
print("✓ Source term evaluation complete!")
print("="*70)
print("\nNext steps:")
print("  - View mms_source_terms_visualization.png for spatial distribution")
print("  - Open mms_source_terms.pvd in ParaView for 3D visualization")
print("  - Check mms_source_terms_nodes.csv for exact nodal values")
print("="*70)
