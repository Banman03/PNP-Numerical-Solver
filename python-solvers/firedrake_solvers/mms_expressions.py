"""
Generated MMS expressions for PNP solver
Use these in pnp_solver_mms.py for verification

Method of Manufactured Solutions:
- Manufactured solutions are defined
- Source terms computed by substituting into strong form
- Add source terms to weak formulation
- Compare FEM solution to exact manufactured solution
"""

from firedrake import *
import numpy as np

# Physical parameters
F = 96485.3329
R = 8.314462618
T = 298.15
F_over_RT = 38.921744810257564
eps = 1.0
D_vals = [1.0, 1.0]
z_vals = [1, -1]
n_species = 2

def get_exact_solutions(mesh, t_val):
    """
    Return exact (manufactured) solutions as Firedrake expressions.
    
    Args:
        mesh: Firedrake mesh
        t_val: current time value
    
    Returns:
        (c_exact_list, phi_exact): tuple of lists and expression
    """
    x, y = SpatialCoordinate(mesh)
    
    c_exact = []
    c_exact.append(1.0 + 0.1*exp(-t_val)*sin(pi*x)*sin(pi*y))  # c_0
    c_exact.append(1.0 + 0.1*exp(-t_val)*sin(pi*x)*sin(pi*y))  # c_1
    
    phi_exact = 0.1*exp(-t_val)*sin(pi*x)*sin(pi*y)
    
    return c_exact, phi_exact

def get_source_terms(mesh, t_val):
    """
    Return source/forcing terms as Firedrake expressions.
    These are added to the RHS of the weak formulation.
    
    Args:
        mesh: Firedrake mesh
        t_val: current time value
    
    Returns:
        (source_c_list, source_phi): tuple of lists and expression
    """
    x, y = SpatialCoordinate(mesh)
    
    source_c = []
    source_c.append((pi**2*(7.78434896205151*exp(t_val)*sin(pi*x)*sin(pi*y) + 1.5568697924103*sin(pi*x)**2*sin(pi*y)**2 - 0.389217448102576*sin(pi*x)**2 - 0.389217448102576*sin(pi*y)**2) + (-0.1 + 0.2*pi**2)*exp(t_val)*sin(pi*x)*sin(pi*y))*exp(-2*t_val))  # S_0
    source_c.append((pi**2*(-7.78434896205151*exp(t_val)*sin(pi*x)*sin(pi*y) - 1.5568697924103*sin(pi*x)**2*sin(pi*y)**2 + 0.389217448102576*sin(pi*x)**2 + 0.389217448102576*sin(pi*y)**2) + (-0.1 + 0.2*pi**2)*exp(t_val)*sin(pi*x)*sin(pi*y))*exp(-2*t_val))  # S_1
    
    source_phi = 0.2*pi**2*exp(-t_val)*sin(pi*x)*sin(pi*y)
    
    return source_c, source_phi

# ============================================================================
# Numpy functions for evaluation at mesh nodes
# ============================================================================

def c0_exact_numpy(x, y, t):
    """Exact solution for c_0"""
    return 1.0 + 0.1*np.exp(-t)*np.sin(np.pi*x)*np.sin(np.pi*y)

def c1_exact_numpy(x, y, t):
    """Exact solution for c_1"""
    return 1.0 + 0.1*np.exp(-t)*np.sin(np.pi*x)*np.sin(np.pi*y)

def phi_exact_numpy(x, y, t):
    """Exact solution for phi"""
    return 0.1*np.exp(-t)*np.sin(np.pi*x)*np.sin(np.pi*y)

def source_c0_numpy(x, y, t):
    """Source term for c_0"""
    return (np.pi**2*(7.78434896205151*np.exp(t)*np.sin(np.pi*x)*np.sin(np.pi*y) + 1.5568697924103*np.sin(np.pi*x)**2*np.sin(np.pi*y)**2 - 0.389217448102576*np.sin(np.pi*x)**2 - 0.389217448102576*np.sin(np.pi*y)**2) + (-0.1 + 0.2*np.pi**2)*np.exp(t)*np.sin(np.pi*x)*np.sin(np.pi*y))*np.exp(-2*t)

def source_c1_numpy(x, y, t):
    """Source term for c_1"""
    return (np.pi**2*(-7.78434896205151*np.exp(t)*np.sin(np.pi*x)*np.sin(np.pi*y) - 1.5568697924103*np.sin(np.pi*x)**2*np.sin(np.pi*y)**2 + 0.389217448102576*np.sin(np.pi*x)**2 + 0.389217448102576*np.sin(np.pi*y)**2) + (-0.1 + 0.2*np.pi**2)*np.exp(t)*np.sin(np.pi*x)*np.sin(np.pi*y))*np.exp(-2*t)

def source_phi_numpy(x, y, t):
    """Source term for phi"""
    return 0.2*np.pi**2*np.exp(-t)*np.sin(np.pi*x)*np.sin(np.pi*y)
