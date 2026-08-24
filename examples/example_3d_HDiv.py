import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from bogovskii import bogovskii_map

np.set_printoptions(precision=2, linewidth=125, suppress=True)


mesh = UnitCubeMesh(10, 10, 10)
(x, y, z) = SpatialCoordinate(mesh)


degree = 3
quadrature_degree = 8

funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree, variant=f"integral({quadrature_degree})")
funspace_RT = FunctionSpace(mesh, "RT", degree, variant=f"integral({quadrature_degree})")

funspace_CG_1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
funspace_Ned1_1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")
funspace_RT_1 = FunctionSpace(mesh, "RT", 1, variant=f"integral({quadrature_degree})")
mixspace_CG_Ned1_1 = MixedFunctionSpace([funspace_CG_1, funspace_Ned1_1])

ndof_Ned1 = funspace_Ned1.finat_element.space_dimension()
ndof_RT = funspace_RT.finat_element.space_dimension()

B = Constant(bogovskii_map(funspace_RT, funspace_Ned1))

domain = f"{{[i,j]: 0 <= i < {ndof_Ned1} and 0 <= j < {ndof_RT}}}"
instructions = f"""
for i
    a[i] = 0.0
    for j
        a[i] = a[i] + T[i*{ndof_RT} + j] * b[j]
    end
end
"""

f = Function(funspace_Ned1).interpolate(as_vector([sin(pi * (1-x-y-z)), sin(pi/2 * (1-x-y-z)), sin(pi/4 * (1-x-y-z))]))
q = Function(funspace_RT).interpolate(curl(f))


p0, u0 = TrialFunctions(mixspace_CG_Ned1_1)
p1, u1 = TestFunctions(mixspace_CG_Ned1_1)
PIq_mixed = Function(mixspace_CG_Ned1_1)

a = (inner(grad(p0), u1)*dx - inner(curl(u0), curl(u1))*dx + inner(p0, p1)*dx - inner(u0, grad(p1))*dx)


Iq = Function(funspace_RT_1).interpolate(q) 
L = inner(Iq, curl(u1))*dx
solve(a==L, PIq_mixed)
_, PIq = PIq_mixed.subfunctions


Mq = Function(funspace_RT).interpolate(q - Iq)
PMq = Function(funspace_Ned1)
par_loop(
    (domain, instructions),
    dx,
    {"T": (B, READ), "b": (Mq, READ), "a": (PMq, WRITE)}
)

#print(Iq.dat.data[:])

print(Mq.dat.data[:])

v = project(curl(PMq), funspace_RT)
print(v.dat.data[:])


Pq = Function(funspace_Ned1).interpolate(PIq + PMq)

print(norm(q - curl(Pq), "L2"))