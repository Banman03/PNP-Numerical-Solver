import itertools

def generate_solver_params():
    snes_types = ['newtonls', 'newtontr', 'ngmres'] 
    line_searches = ['bt', 'l2', 'cp', 'bisection']
    
    ksp_pc_pairs = [
        {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        {'ksp_type': 'gmres', 'pc_type': 'ilu', 'ksp_rtol': 1e-7},
        {'ksp_type': 'gmres', 'pc_type': 'fieldsplit', 'pc_fieldsplit_type': 'additive'}
    ]

    combinations = []
    
    for snes, ls, kp in itertools.product(snes_types, line_searches, ksp_pc_pairs):
        params = {
            'snes_type': snes,
            'snes_max_it': 100,
            'snes_atol': 1e-8,
            'snes_rtol': 1e-8,
        }
        
        if snes != 'newtontr':
            params['snes_linesearch_type'] = ls
            
        params.update(kp)
        
        if snes == 'newtontr' and ls != line_searches[0]:
            continue
            
        combinations.append(params)
        
    return combinations