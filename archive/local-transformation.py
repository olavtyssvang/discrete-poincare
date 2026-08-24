import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.parloops import par_loop, READ, WRITE

mesh = UnitTriangleMesh(3)
x, y = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "N1curl", 2)

u = Function(V).interpolate(as_vector([sin(x)*cos(y), cos(x)*sin(y)]))
v = Function(V)

n = V.finat_element.space_dimension()   # local cell dof count

M = Constant(np.eye(n))

domain = f"{{[i,j]: 0 <= i < {n} and 0 <= j < {n}}}"
instructions = f"""
for i
  v[i] = 0.0
  for j
    v[i] = v[i] + M[i*{n} + j] * u[j]
  end
end
"""

par_loop(
    (domain, instructions),
    dx,
    {"u": (u, READ), "v": (v, WRITE), "M": (M, READ)}
)

fig, axes = plt.subplots(1, 2)
quiver(u, axes=axes[0])
quiver(v, axes=axes[1])
plt.show()