# docker start -ai eloquent_hypatia

from dolfin import *
from dolfin_adjoint import *

import numpy as np
from math import sqrt


epsilon0  = 8.8541878128e-12
e0        = 1.60217663e-19
N_A       = 6.0221408e+23
kB        = 1.380649e-23
pot       = -1.200000
T         = 2.980000000000e+02
pzc       = 0.000000
potential = (pot - pzc)* e0 / kB / T
delta_RP  = 3.200000000000e-10
epsilon_RP= 4.170000
epsilon_s = 78.500000

n0        = N_A
D0        = 1.0e-9
lambda_d  = np.sqrt(epsilon_s * epsilon0 * kB * T / e0**2 / n0)
Ld        = lambda_d * 60
factor_a  = delta_RP*epsilon_s/epsilon_RP/lambda_d
factor_1  = Ld/lambda_d
factor_2  = Ld*lambda_d/D0
ns_ions   = 2

Length_limit = Ld / lambda_d

pi = np.pi


increments = [64, 128, 256]
solutions = []

for N in increments:

    domain = RectangleMesh(Point(0.0, 0.0), Point(Length_limit, Length_limit), N, N)

    V = VectorFunctionSpace(domain, "Lagrange", 1, dim=ns_ions+1)


        
    x = SpatialCoordinate(domain)


                        
    dt = Constant(1.0e-3)        
    t = Constant(0.0)
    # t_n = Constant(1.0e-3)        

    c1 = 1.0 + 0.5 * sin(pi * x[0] / Length_limit) * sin(pi * x[1] / Length_limit)
    c2  = 1.0 - 0.5 * sin(pi * x[0] / Length_limit) * sin(pi * x[1] / Length_limit)
    phi= sin(pi * x[0] / Length_limit) * sin(pi * x[1] / Length_limit)

    # c1_n  = 1.0 + 0.5 * sin(t_n) * sin(pi * x[0] / Length_limit) * sin(pi * x[1] / Length_limit)
    # c2_n  = 1.0 - 0.5 * sin(t_n) * sin(pi * x[0] / Length_limit) * sin(pi * x[1] / Length_limit)
    # phi_n= sin(t_n) * sin(pi * x[0] / Length_limit) * sin(pi * x[1] / Length_limit)







    dotc1 = inner(grad(c1), grad(phi))
    dotc2 = inner(grad(c2), grad(phi))

    d1 = Constant(9.311000000000)
    d2 = Constant(5.273000000000)

    s1 = -factor_1 * d1 * (div(grad(c1)) + (+1)*dotc1 + (+1)*c1* div(grad(phi)))
    s2 = -factor_1 * d2 * (div(grad(c2)) + (-1)*dotc2 + (-1)*c2* div(grad(phi)))

    p  =  div(grad(phi)) - ((+1)*c1 + (-1)*c2)





    tol = 1e-8
    right_boundary = lambda x, on_boundary: on_boundary and near(x[0], Length_limit, tol)

    bc_H   = DirichletBC(V.sub(0), Constant(1.0), "on_boundary")
    bc_OH  = DirichletBC(V.sub(1), Constant(1.0), "on_boundary")
    bc_phi = DirichletBC(V.sub(2), Constant(0.0), "on_boundary")
    bcs = [bc_H, bc_OH, bc_phi]


    # u_n = Function(V)
    u   = Function(V)

    v  = TestFunction(V)
    du = TrialFunction(V)

    # u_n_H, u_n_OH, u_n_phi = u_n[0], u_n[1], u_n[2]
    u_H, u_OH, u_phi   = u[0], u[1], u[2]
    v_H, v_OH, v_phi   = v[0], v[1], v[2]

   


    u.interpolate(Expression(
    ("1.0 + 0.5*sin(pi*x[0]/L)*sin(pi*x[1]/L)",
     "1.0 - 0.5*sin(pi*x[0]/L)*sin(pi*x[1]/L)",
     "sin(pi*x[0]/L)*sin(pi*x[1]/L)"),
    degree=4, pi=pi, L=Length_limit))




    flux_OH_at_left = Constant(1.0e-5)
    dt = Constant(1.0e-3)



    flux_H = 9.311000000000e+00*u_H*(1.0)*grad(u_phi)+9.311000000000e+00*grad(u_H)
    flux_OH = 5.273000000000e+00*u_OH*(-1.0)*grad(u_phi)+5.273000000000e+00*grad(u_OH)

    F_H = factor_1*dot(flux_H, grad(v_H))*dx + s1*v_H*dx
    F_OH = factor_1*dot(flux_OH, grad(v_OH))*dx + s2*v_OH*dx
    F_phi = (u_phi - potential)/factor_a*v_phi*ds - dot(grad(u_phi), grad(v_phi))*dx + ((1.0)*u_H+(-1.0)*u_OH)*v_phi*dx - p*v_phi*dx
    F = F_H + F_OH + F_phi


    J = derivative(F, u, du)

    problem = NonlinearVariationalProblem(F, u, bcs, J)
    solver  = NonlinearVariationalSolver(problem)

    prm = solver.parameters["newton_solver"]
    prm["convergence_criterion"] = "residual"
    prm["relative_tolerance"]    = 1e-6
    prm["linear_solver"]         = "lu"

    PETScOptions.set("ksp_type", "preonly")
    PETScOptions.set("pc_type", "lu")
    PETScOptions.set("pc_factor_mat_solver_type", "mumps")

    solver.solve()
    solutions.append((V, u))


    H_L2  = sqrt(assemble(((u[0] - c1)**2) * dx))
    OH_L2 = sqrt(assemble(((u[1] - c2)**2) * dx))
    phi_L2  = sqrt(assemble(((u[2] - phi)**2) * dx))

    # 
    print(f"N={N} | L2:  H={H_L2:.3e}  OH={OH_L2:.3e}  U={phi_L2:.3e} ")

