import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from bogovskii import bogovskii_map

np.set_printoptions(precision=2, linewidth=125, suppress=True)


mesh = UnitCubeMesh(10, 10, 10)
(x, y, z) = SpatialCoordinate(mesh)


degree = 3
quadrature_degree = 8

funspace_CG = FunctionSpace(mesh, "CG", degree + 1, variant="equispaced")
funspace_Ned2 = FunctionSpace(mesh, "N2curl", degree, variant=f"integral({quadrature_degree})")

funspace_CG_1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
funspace_Ned1_1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")

ndof_CG = funspace_CG.finat_element.space_dimension()
ndof_Ned = funspace_Ned2.finat_element.space_dimension()

B = Constant(bogovskii_map(funspace_Ned2, funspace_CG))

domain = f"{{[i,j]: 0 <= i < {ndof_CG} and 0 <= j < {ndof_Ned}}}"
instructions = f"""
for i
    a[i] = 0.0
    for j
        a[i] = a[i] + T[i*{ndof_Ned} + j] * b[j]
    end
end
"""

f = Function(funspace_CG).interpolate(sin(pi * (1-x-y-z)))
u = Function(funspace_Ned2).interpolate(grad(f))


p0 = TrialFunction(funspace_CG_1)
p1 = TestFunction(funspace_CG_1)
PIu = Function(funspace_CG_1)

a = inner(grad(p0), grad(p1))*dx
nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)


Iu = Function(funspace_Ned1_1).interpolate(u) 
L = inner(Iu, grad(p1))*dx
solve(a==L, PIu, nullspace=nullspace)

Mu = Function(funspace_Ned2).interpolate(u - Iu)
PMu = Function(funspace_CG)
par_loop(
    (domain, instructions),
    dx,
    {"T": (B, READ), "b": (Mu, READ), "a": (PMu, WRITE)}
)


Pu = Function(funspace_CG).interpolate(PIu + PMu)

print(norm(u - grad(Pu), "L2"))