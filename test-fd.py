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

funspace_Ned1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")

ndof_CG = funspace_CG.finat_element.space_dimension()
ndof_Ned = funspace_Ned.finat_element.space_dimension()

B = Constant(bogovskii.map(degree, "face"))

domain = f"{{[i,j]: 0 <= i < {ndof_CG} and 0 <= j < {ndof_Ned}}}"
instructions = f"""
for i
  p[i] = 0.0
  for j
    p[i] = p[i] + B[i*{ndof_Ned} + j] * u[j]
  end
end
"""

for bc in [1-x-y]:
    p0 = Function(funspace_CG).interpolate(sin(pi * bc))
    u0 = Function(funspace_Ned).interpolate(grad(p0))

    Iu = Function(funspace_Ned1).interpolate(u0) 
    Mu = Function(funspace_Ned).interpolate(u0 - Iu)
    #u = Function(funspace_Ned).interpolate(grad(x * y * y))
    p = Function(funspace_CG)

    par_loop(
        (domain, instructions),
        dx,
        {"u": (Mu, READ), "p": (p, WRITE), "B": (B, READ)}
    )

    print(norm(Mu - grad(p), "L2"))

    n_axes = 3
    fig, axes = plt.subplots(1, n_axes, figsize=(18, 5))
    for i in range(n_axes):
        axes[i].axis("equal")

    tripcolor(p, axes=axes[0], cmap="viridis")
    quiver(grad(p), axes=axes[1])
    quiver(Mu, axes=axes[2])
    plt.show()