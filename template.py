# https://fenicsproject.discourse.group/t/how-to-create-a-source-term-that-is-a-spatial-function/11074
# optimization that works well with fem?

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

domain = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, points=[(0.0, 0.0), (Length_limit, Length_limit)], n = (32, 32))
element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(ns_ions+1,))

V = dolfinx.fem.functionspace(domain, element)

fdim = domain.topology.dim - 1


x = ufl.SpatialCoordinate(domain)


c1 = 1 + 0.5 * ufl.sin(ufl.pi * x[0] / Length_limit) * ufl.sin(ufl.pi * x[1] / Length_limit)
c2  = 1 - 0.5 * ufl.sin(ufl.pi * x[0] / Length_limit) * ufl.sin(ufl.pi * x[1] / Length_limit)
phi= ufl.sin(ufl.pi * x[0] / Length_limit) * ufl.sin(ufl.pi * x[1] / Length_limit)


d1 = 9.311000000000e+00
d2 = 5.273000000000e+00



dotc1 = ufl.inner(ufl.grad(c1), ufl.grad(phi))
dotc2 = ufl.inner(ufl.grad(c2), ufl.grad(phi))
s1 = factor_1 * d1 * (ufl.div(ufl.grad(c1)) + 1*dotc1 + 1*c1*ufl.div(ufl.grad(phi)))
s2 = factor_1 * d2 * (ufl.div(ufl.grad(c2)) - 1*dotc2 - 1*c2*ufl.div(ufl.grad(phi)))
p  = ufl.div(ufl.grad(phi)) - (1*c1 - 1*c2)



right_boundary = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(Length_limit, x[0]))
left_boundary = dolfinx.mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(0.0, x[0]))


V0, _ = V.sub(0).collapse()
V1, _ = V.sub(1).collapse()
V2, _ = V.sub(2).collapse()

gH = dolfinx.fem.Function(V0)
gO = dolfinx.fem.Function(V1)
gP = dolfinx.fem.Function(V2)

gH.interpolate(lambda X: 1 + 0.5*np.sin(np.pi*X[0]/Length_limit)*np.sin(np.pi*X[1]/Length_limit))
gO.interpolate(lambda X: 1 - 0.5*np.sin(np.pi*X[0]/Length_limit)*np.sin(np.pi*X[1]/Length_limit))
gP.interpolate(lambda X: np.sin(np.pi*X[0]/Length_limit)*np.sin(np.pi*X[1]/Length_limit))


# bc_H   = dolfinx.fem.dirichletbc(gH, dolfinx.fem.locate_dofs_topological(V.sub(0), fdim, right_boundary))
# bc_OH  = dolfinx.fem.dirichletbc(gO, dolfinx.fem.locate_dofs_topological(V.sub(1), fdim, right_boundary))
# bc_phi = dolfinx.fem.dirichletbc(gP, dolfinx.fem.locate_dofs_topological(V.sub(2), fdim, right_boundary))


bc_H = dolfinx.fem.dirichletbc(PETSc.ScalarType(1.000000000000e-04), 
                              dolfinx.fem.locate_dofs_topological(V.sub(0), fdim, right_boundary), 
                              V.sub(0))

bc_OH = dolfinx.fem.dirichletbc(PETSc.ScalarType(1.000000000000e-04), 
                              dolfinx.fem.locate_dofs_topological(V.sub(1), fdim, right_boundary), 
                              V.sub(1))

bc_phi = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.000000000000e+00), 
                              dolfinx.fem.locate_dofs_topological(V.sub(2), fdim, right_boundary), 
                              V.sub(2))


u_n = dolfinx.fem.Function(V) 
u = dolfinx.fem.Function(V)  
u_n_H, u_n_OH, u_n_phi = ufl.split(u_n)
u_H, u_OH, u_phi = ufl.split(u)
v_H, v_OH, v_phi = ufl.TestFunction(V)

u_n.sub(0).interpolate(lambda x: 1.000000000000e-04*np.ones(x[0].shape))
u_n.sub(1).interpolate(lambda x: 1.000000000000e-04*np.ones(x[0].shape))
u_n.sub(2).interpolate(lambda x: 7.944060176975e-03*x[0]+(-4.672608459440e+01))
u_n.x.scatter_forward()


flux_OH_at_left = dolfinx.fem.Constant(domain, 1.0e-5)
dt = dolfinx.fem.Constant(domain, 1.0e-10)
flux_H  = d1*( u_H*(+1.0)*ufl.grad(u_phi) + ufl.grad(u_H) )
flux_OH = d2*( u_OH*(-1.0)*ufl.grad(u_phi) + ufl.grad(u_OH) )

F_H   =  factor_1*ufl.dot(flux_H,  ufl.grad(v_H))*ufl.dx  -  s1*v_H*ufl.dx
F_OH  =  factor_1*ufl.dot(flux_OH, ufl.grad(v_OH))*ufl.dx -  s2*v_OH*ufl.dx
F_phi =  ufl.dot(ufl.grad(u_phi), ufl.grad(v_phi))*ufl.dx - ( (+1.0)*u_H + (-1.0)*u_OH )*v_phi*ufl.dx - p*v_phi*ufl.dx

F = F_H + F_OH + F_phi

problem = NonlinearProblem(F, u, bcs=[bc_H, bc_OH, bc_phi])

print("before Newton solver")

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-6

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "cg"
# opts[f"{option_prefix}pc_type"] = "gamg"
# opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

n, converged = solver.solve(u)
print(converged)

H_L2  = ufl.sqrt(ufl.assemble(((u[0] - c1)**2) * ufl.dx))
OH_L2 = ufl.sqrt(ufl.assemble(((u[1] - c2)**2) * ufl.dx))
phi_L2  = ufl.sqrt(ufl.assemble(((u[2] - phi)**2) * ufl.dx))
 
print(f"L2:  H={H_L2:.3e}  OH={OH_L2:.3e}  U={phi_L2:.3e} ")

