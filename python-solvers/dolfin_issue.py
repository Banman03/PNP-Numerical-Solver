# -*- coding:utf-8 -*-
import ufl 
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np 
import basix

epsilon0  = 8.8541878128e-12 #vacuum permittivity, F/m 
e0        = 1.60217663e-19 #electron charge 
N_A       = 6.0221408e+23 #Avogadro constant. 
kB        = 1.380649e-23 #m^2 kg s^(-2) K^(-1) 
pot       = -1.200000 #electrode potential
T         = 2.980000000000e+02 #temperature
pzc       = 0.000000 #potential of zero charge
potential = (pot - pzc)* e0 / kB / T 
delta_RP  = 3.200000000000e-10 #the thickness of Helmholtz layer
epsilon_RP= 4.170000 #the dielectric constant of Helmholtz layer
epsilon_s = 78.500000 #the dielectric constant of bulk solution

n0        = N_A
D0        = 1.0e-9
lambda_d  = np.sqrt(epsilon_s * epsilon0 * kB * T / e0**2 / n0)
Ld        = lambda_d * 60 #the position of bulk
factor_a  = delta_RP*epsilon_s/epsilon_RP/lambda_d
factor_1  = Ld/lambda_d
factor_2  = Ld*lambda_d/D0
ns_ions   = 2

Length_limit = Ld / lambda_d

domain = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 2048, points=(0.0, Length_limit))
element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(ns_ions+1,))

V = dolfinx.fem.functionspace(domain, element)

fdim = domain.topology.dim - 1

right_boundary = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(Length_limit, x[0]))
left_boundary = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(0.0, x[0]))

bc_H = dolfinx.fem.dirichletbc(PETSc.ScalarType(1.000000000000e-04), 
                              dolfinx.fem.locate_dofs_topological(V.sub(0), fdim, right_boundary), 
                              V.sub(0))

bc_OH = dolfinx.fem.dirichletbc(PETSc.ScalarType(1.000000000000e-04), 
                              dolfinx.fem.locate_dofs_topological(V.sub(1), fdim, right_boundary), 
                              V.sub(1))

bc_phi = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.000000000000e+00), 
                              dolfinx.fem.locate_dofs_topological(V.sub(2), fdim, right_boundary), 
                              V.sub(2))
u_n = dolfinx.fem.Function(V) #the solution of previous step.
u = dolfinx.fem.Function(V)   #the solution need to solve at the current step.
u_n_H, u_n_OH, u_n_phi = ufl.split(u_n)
u_H, u_OH, u_phi = ufl.split(u)
v_H, v_OH, v_phi = ufl.TestFunction(V)

#set the initial condition
u_n.sub(0).interpolate(lambda x: 1.000000000000e-04*np.ones(x[0].shape))
u_n.sub(1).interpolate(lambda x: 1.000000000000e-04*np.ones(x[0].shape))
#initial value for the potential 
u_n.sub(2).interpolate(lambda x: 7.944060176975e-03*x[0]+(-4.672608459440e+01))
u_n.x.scatter_forward()


#Define the constants for the flux.
flux_OH_at_left = dolfinx.fem.Constant(domain, 1.0e-5)
dt = dolfinx.fem.Constant(domain, 1.0e-10)

flux_H = 9.311000000000e+00*u_H*(1.0)*ufl.grad(u_n_phi)+9.311000000000e+00*ufl.grad(u_H)
flux_OH = 5.273000000000e+00*u_OH*(-1.0)*ufl.grad(u_n_phi)+5.273000000000e+00*ufl.grad(u_OH)

F_H = u_H*v_H*ufl.dx + dt*factor_1*ufl.dot(flux_H,ufl.grad(v_H))*ufl.dx -u_n_H*v_H*ufl.dx
F_OH = u_OH*v_OH*ufl.dx + dt*factor_1*ufl.dot(flux_OH,ufl.grad(v_OH))*ufl.dx -dt*factor_1*flux_OH_at_left*v_OH*ufl.ds -u_n_OH*v_OH*ufl.dx
F_phi = (u_phi - potential)/factor_a*v_phi*ufl.ds - ufl.dot(ufl.grad(u_phi), ufl.grad(v_phi))*ufl.dx + ((1.0)*u_H+(-1.0)*u_OH)*v_phi*ufl.dx

F = F_H + F_OH + F_phi

problem = NonlinearProblem(F, u, bcs=[bc_H, bc_OH, bc_phi])

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-6

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

n, converged = solver.solve(u)