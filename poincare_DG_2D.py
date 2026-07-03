import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
import bogovskii

#np.set_printoptions(precision=2, linewidth=125, suppress=True)


mesh = UnitTriangleMesh(0)
(u, v) = SpatialCoordinate(mesh)
x = u
y = v 
V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=2)
coords = Function(V_coord)
coords.interpolate(as_vector([x, y]))
mesh = Mesh(coords)
(x, y) = SpatialCoordinate(mesh)


degree = 2
quadrature_degree = 8

funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant=f"integral({quadrature_degree})")
funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant=f"integral({quadrature_degree})")

funspace_CG1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
funspace_Ned1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")
funspace_DG0 = FunctionSpace(mesh, "DG", 0, variant=f"integral({quadrature_degree})")
mixspace_CG1_Ned1 = MixedFunctionSpace([funspace_CG1, funspace_Ned1])

ndof_Ned = funspace_Ned.finat_element.space_dimension()
ndof_DG = funspace_DG.finat_element.space_dimension()

B = Constant(bogovskii._map_DG_face(degree))

domain = f"{{[i,j]: 0 <= i < {ndof_Ned} and 0 <= j < {ndof_DG}}}"
instructions = f"""
for i
    a[i] = 0.0
    for j
        a[i] = a[i] + T[i*{ndof_DG} + j] * b[j]
    end
end
"""

f = Function(funspace_Ned).interpolate(as_vector([sin(pi * (1-x-y)), sin(pi/2 * (1-x-y))]))
q = Function(funspace_DG).interpolate(curl(f))


p0, u0 = TrialFunctions(mixspace_CG1_Ned1)
p1, u1 = TestFunctions(mixspace_CG1_Ned1)
PIq_mixed = Function(mixspace_CG1_Ned1)

a = (inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx + inner(p0, p1)*dx - inner(u0, grad(p1))*dx)


Iq = Function(funspace_DG0).interpolate(q) 
L = inner(Iq, curl(u1))*dx
solve(a==L, PIq_mixed)
_, PIq = PIq_mixed.subfunctions

Mq = Function(funspace_DG).interpolate(q - Iq)
PMq = Function(funspace_Ned)
par_loop(
    (domain, instructions),
    dx,
    {"T": (B, READ), "b": (Mq, READ), "a": (PMq, WRITE)}
)

print(Iq.dat.data[:])
print(Mq.dat.data[:])

Pq = Function(funspace_Ned).interpolate(PIq + PMq)

print(norm(Iq - curl(PIq), "L2"))
print(norm(Mq - curl(PMq), "L2"))

n_axes = 3
fig, axes = plt.subplots(1, n_axes, figsize=(18, 5))
for i in range(n_axes):
    axes[i].axis("equal")

quiver(PMq, axes=axes[0])
tripcolor(project(curl(PMq), funspace_DG), axes=axes[1], cmap="viridis")
tripcolor(Mq, axes=axes[2], cmap="viridis")
plt.show()