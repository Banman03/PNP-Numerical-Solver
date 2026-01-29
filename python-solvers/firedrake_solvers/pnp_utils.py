from firedrake import *
import itertools

def generate_solver_params():
    """
    Generates a list of solver parameter dictionaries for testing 
    convergence on PNP equations.
    """
    snes_types = ['newtonls', 'newtonal'] 
    
    line_searches = ['bt', 'l2', 'basic']
    
    # 3. Linear Solver (KSP) and Preconditioner (PC) pairs
    # Direct solvers are more robust but also memory intensive; 
    # Iterative solvers (GMRES) are faster but need good PCs.
    ksp_pc_pairs = [
        {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        {'ksp_type': 'gmres', 'pc_type': 'ilu', 'ksp_rtol': 1e-7},
        {'ksp_type': 'gmres', 'pc_type': 'fieldsplit', 'pc_fieldsplit_type': 'additive'}
    ]

    combinations = []
    
    for snes, ls, kp in itertools.product(snes_types, line_searches, ksp_pc_pairs):
        params = {
            'snes_type': snes,
            'snes_linesearch_type': ls,
            'snes_max_it': 100,
            'snes_atol': 1e-8,
            'snes_rtol': 1e-8,
            'snes_monitor': None,
        }
        params.update(kp)
        combinations.append(params)
        
    return combinations