import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from bogovskii import bogovskii_map

np.set_printoptions(precision=2, linewidth=125, suppress=True)


mesh = UnitSquareMesh(10, 10)
(x, y) = SpatialCoordinate(mesh)


degree = 3
quadrature_degree = 8

funspace_RT = FunctionSpace(mesh, "RT", degree, variant=f"integral({quadrature_degree})")
funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="spectral")

funspace_CG_1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
funspace_RT_1 = FunctionSpace(mesh, "RT", 1, variant=f"integral({quadrature_degree})")
funspace_DG_0 = FunctionSpace(mesh, "DG", 0, variant=f"integral({quadrature_degree})")
mixspace_CG_RT_1 = MixedFunctionSpace([funspace_CG_1, funspace_RT_1])

ndof_RT = funspace_RT.finat_element.space_dimension()
ndof_DG = funspace_DG.finat_element.space_dimension()

B = Constant(bogovskii_map(funspace_DG, funspace_RT))

domain = f"{{[i,j]: 0 <= i < {ndof_RT} and 0 <= j < {ndof_DG}}}"
instructions = f"""
for i
    a[i] = 0.0
    for j
        a[i] = a[i] + T[i*{ndof_DG} + j] * b[j]
    end
end
"""

f = Function(funspace_RT).interpolate(as_vector([sin(pi * (1-x-y)), sin(pi/2 * (1-x-y))]))
q = Function(funspace_DG).interpolate(div(f))


p0, u0 = TrialFunctions(mixspace_CG_RT_1)
p1, u1 = TestFunctions(mixspace_CG_RT_1)
PIq_mixed = Function(mixspace_CG_RT_1)

a = (inner(curl(p0), u1)*dx + inner(div(u0), div(u1))*dx + inner(p0, p1)*dx - inner(u0, curl(p1))*dx)


Iq = Function(funspace_DG_0).interpolate(q) 
L = inner(Iq, div(u1))*dx
solve(a==L, PIq_mixed)
_, PIq = PIq_mixed.subfunctions


Mq = Function(funspace_DG).interpolate(q - Iq)
Mq_scaled = Function(funspace_DG).interpolate(JacobianDeterminant(mesh) * Mq)
PMq = Function(funspace_RT)
par_loop(
    (domain, instructions),
    dx,
    {"T": (B, READ), "b": (Mq_scaled, READ), "a": (PMq, WRITE)}
)

#print(Iq.dat.data[:])

print(Mq.dat.data[:])

v = project(div(PMq), funspace_DG)
print(v.dat.data[:])


Pq = Function(funspace_RT).interpolate(PIq + PMq)

print(norm(q - div(Pq), "L2"))

n_axes = 3
fig, axes = plt.subplots(1, n_axes, figsize=(18, 5))
for i in range(n_axes):
    axes[i].axis("equal")

quiver(Pq, axes=axes[0])
tripcolor(project(div(Pq), funspace_DG), axes=axes[1], cmap="viridis")
tripcolor(q, axes=axes[2], cmap="viridis")
plt.show()