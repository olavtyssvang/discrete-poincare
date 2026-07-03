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

ndof_CG = funspace_CG.finat_element.space_dimension()
ndof_Ned = funspace_Ned.finat_element.space_dimension()

B = Constant(bogovskii._map_Ned_face(degree))

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
u = Function(funspace_Ned).interpolate(grad(f))


p0 = TrialFunction(funspace_CG1)
p1 = TestFunction(funspace_CG1)
PIu = Function(funspace_CG1)

a = inner(grad(p0), grad(p1))*dx
nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)


Iu = Function(funspace_Ned1).interpolate(u) 
L = inner(Iu, grad(p1))*dx
solve(a==L, PIu, nullspace=nullspace)

Mu = Function(funspace_Ned).interpolate(u - Iu)
PMu = Function(funspace_CG)
par_loop(
    (domain, instructions),
    dx,
    {"T": (B, READ), "b": (Mu, READ), "a": (PMu, WRITE)}
)


Pu = Function(funspace_CG).interpolate(PIu + PMu)

print(norm(u - grad(Pu), "L2"))

n_axes = 3
fig, axes = plt.subplots(1, n_axes, figsize=(18, 5))
for i in range(n_axes):
    axes[i].axis("equal")

tripcolor(Pu, axes=axes[0], cmap="viridis")
quiver(grad(Pu), axes=axes[1])
quiver(u, axes=axes[2])
plt.show()