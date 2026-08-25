import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from bogovskii import bogovskii_map

#np.set_printoptions(precision=2, linewidth=125, suppress=True)


mesh = UnitSquareMesh(10, 10)
(x, y) = SpatialCoordinate(mesh)


degree = 3
quadrature_degree = 8

funspace_Ned2 = FunctionSpace(mesh, "N2curl", degree, variant=f"integral({quadrature_degree})")
funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="spectral")

funspace_CG_1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
funspace_Ned1_1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")
funspace_DG_0 = FunctionSpace(mesh, "DG", 0, variant=f"integral({quadrature_degree})")
mixspace_CG_Ned1_1 = MixedFunctionSpace([funspace_CG_1, funspace_Ned1_1])

ndof_Ned2 = funspace_Ned2.finat_element.space_dimension()
ndof_DG = funspace_DG.finat_element.space_dimension()

B = Constant(bogovskii_map(funspace_DG, funspace_Ned2))

domain = f"{{[i,j]: 0 <= i < {ndof_Ned2} and 0 <= j < {ndof_DG}}}"
instructions = f"""
for i
    a[i] = 0.0
    for j
        a[i] = a[i] + T[i*{ndof_DG} + j] * b[j]
    end
end
"""

f = Function(funspace_Ned2).interpolate(as_vector([sin(pi * (1-x-y)), sin(pi/2 * (1-x-y))]))
q = Function(funspace_DG).interpolate(curl(f))


p0, u0 = TrialFunctions(mixspace_CG_Ned1_1)
p1, u1 = TestFunctions(mixspace_CG_Ned1_1)
PIq_mixed = Function(mixspace_CG_Ned1_1)

a = (inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx + inner(p0, p1)*dx - inner(u0, grad(p1))*dx)


Iq = Function(funspace_DG_0).interpolate(q) 
L = inner(Iq, curl(u1))*dx
solve(a==L, PIq_mixed)
_, PIq = PIq_mixed.subfunctions


Mq = Function(funspace_DG).interpolate(q - Iq)
Mq_scaled = Function(funspace_DG).interpolate(JacobianDeterminant(mesh) * Mq)
PMq = Function(funspace_Ned2)
par_loop(
    (domain, instructions),
    dx,
    {"T": (B, READ), "b": (Mq_scaled, READ), "a": (PMq, WRITE)}
)

#print(Iq.dat.data[:])

print(Mq.dat.data[:])

v = project(curl(PMq), funspace_DG)
print(v.dat.data[:])


Pq = Function(funspace_Ned2).interpolate(PIq + PMq)

print(norm(q - curl(Pq), "L2"))

n_axes = 3
fig, axes = plt.subplots(1, n_axes, figsize=(18, 5))
for i in range(n_axes):
    axes[i].axis("equal")

quiver(Pq, axes=axes[0])
tripcolor(project(curl(Pq), funspace_DG), axes=axes[1], cmap="viridis")
tripcolor(q, axes=axes[2], cmap="viridis")
plt.show()