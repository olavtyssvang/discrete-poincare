import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from bogovskii import bogovskii_map

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

funspace_CG = FunctionSpace(mesh, "CG", degree + 1, variant="equispaced")
funspace_BDM = FunctionSpace(mesh, "BDM", degree, variant=f"integral({quadrature_degree})")

funspace_CG_1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
funspace_RT_1 = FunctionSpace(mesh, "RT", 1, variant=f"integral({quadrature_degree})")

ndof_CG = funspace_CG.finat_element.space_dimension()
ndof_Ned = funspace_BDM.finat_element.space_dimension()

B = Constant(bogovskii_map(funspace_BDM, funspace_CG))

domain = f"{{[i,j]: 0 <= i < {ndof_CG} and 0 <= j < {ndof_Ned}}}"
instructions = f"""
for i
    a[i] = 0.0
    for j
        a[i] = a[i] + T[i*{ndof_Ned} + j] * b[j]
    end
end
"""

f = Function(funspace_CG).interpolate(sin(pi * (1-x-y)))
u = Function(funspace_BDM).interpolate(curl(f))


p0 = TrialFunction(funspace_CG_1)
p1 = TestFunction(funspace_CG_1)
PIu = Function(funspace_CG_1)

a = inner(curl(p0), curl(p1))*dx
nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)


Iu = Function(funspace_RT_1).interpolate(u) 
L = inner(Iu, curl(p1))*dx
solve(a==L, PIu, nullspace=nullspace)

Mu = Function(funspace_BDM).interpolate(u - Iu)
PMu = Function(funspace_CG)
par_loop(
    (domain, instructions),
    dx,
    {"T": (B, READ), "b": (Mu, READ), "a": (PMu, WRITE)}
)


Pu = Function(funspace_CG).interpolate(PIu + PMu)

print(norm(u - curl(Pu), "L2"))

n_axes = 3
fig, axes = plt.subplots(1, n_axes, figsize=(18, 5))
for i in range(n_axes):
    axes[i].axis("equal")

tripcolor(Pu, axes=axes[0], cmap="viridis")
quiver(curl(Pu), axes=axes[1])
quiver(u, axes=axes[2])
plt.show()