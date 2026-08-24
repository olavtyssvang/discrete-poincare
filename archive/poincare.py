import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
import bogovskii

np.set_printoptions(precision=2, linewidth=125, suppress=True)


mesh = UnitTriangleMesh(3)
(u, v) = SpatialCoordinate(mesh)
x = u + 1/2 * v
y = sqrt(3)/2 * v 
V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=2)
coords = Function(V_coord)
coords.interpolate(as_vector([x, y]))
mesh = Mesh(coords)
(x, y) = SpatialCoordinate(mesh)


degree = 3
quadrature_degree = 8

funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant=f"integral({quadrature_degree})")

funspace_CG1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
funspace_Ned1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")

f = Function(funspace_CG).interpolate(sin(pi * (1-x-y)))
u = Function(funspace_Ned).interpolate(grad(f))


PIu_data = Function(funspace_CG1)
PIu = TrialFunction(funspace_CG1)
p = TestFunction(funspace_CG1)

a = inner(grad(PIu), grad(p))*dx
nullspace = VectorSpaceBasis(constant=True)


PMu = Function(funspace_CG)
B = Constant(bogovskii.map(degree, "face"))

ndof_CG = funspace_CG.finat_element.space_dimension()
ndof_Ned = funspace_Ned.finat_element.space_dimension()

domain = f"{{[i,j]: 0 <= i < {ndof_CG} and 0 <= j < {ndof_Ned}}}"
instructions = f"""
for i
  PMu[i] = 0.0
  for j
      PMu[i] = PMu[i] + B[i*{ndof_Ned} + j] * Mu[j]
  end
end
"""


Iu = Function(funspace_Ned1).interpolate(u)
Mu = Function(funspace_Ned).interpolate(u - Iu)

L = inner(Iu, grad(p))*dx
solve(a==L, PIu_data, nullspace=nullspace)

par_loop(
    (domain, instructions),
    dx,
    {"Mu": (Mu, READ), "PMu": (PMu, WRITE), "B": (B, READ)}
)


Pu = Function(funspace_CG).interpolate(PIu_data + PMu)

print(norm(grad(PIu_data) - Iu, "L2"))
print(norm(grad(PMu) - Mu, "L2"))
print(norm(grad(Pu) - u, "L2"))