import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
import csv, time
import bogovskii
import gc


MIN_N_SOLVES = 10
MIN_RUNTIME_S = 1

experiment_degrees = [2, 3, 4]
experiment_n_refinements = 5


base_mesh = UnitTriangleMesh(3)
(u, v) = SpatialCoordinate(base_mesh)
x = u + 1/2 * v
y = sqrt(3)/2 * v 
V_coord = VectorFunctionSpace(base_mesh, "CG", 1, dim=2)
coords = Function(V_coord)
coords.interpolate(as_vector([x, y]))
base_mesh = Mesh(coords)

quadrature_degree = 6


def runtime_naive(mesh, degree):
    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant=f"integral({quadrature_degree})")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant=f"integral({quadrature_degree})")
    mixspace_CG_Ned = MixedFunctionSpace([funspace_CG, funspace_Ned])

    (x, y) = SpatialCoordinate(mesh)
    f = Function(funspace_Ned).interpolate(as_vector([sin(pi * (1-x-y)), sin(pi/2 * (1-x-y))]))
    q = Function(funspace_DG).interpolate(curl(f))


    p0, u0 = TrialFunctions(mixspace_CG_Ned)
    p1, u1 = TestFunctions(mixspace_CG_Ned)
    Pq_mixed = Function(mixspace_CG_Ned)
    Pq = Function(funspace_Ned)

    a = (inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx + inner(p0, p1)*dx - inner(u0, grad(p1))*dx)


    time_start = time.time()
    
    n_solves = 0
    while True:
        for i in range(MIN_N_SOLVES):
            L = inner(q, curl(u1))*dx
            solve(a==L, Pq_mixed)
            _, Pq = Pq_mixed.subfunctions
        
        n_solves += MIN_N_SOLVES
        if time.time() - time_start > MIN_RUNTIME_S:
            break

    time_elapsed = time.time() - time_start

    assert norm(curl(Pq) - q, "L2") < 1e-10

    return time_elapsed, n_solves


def runtime_local(mesh, degree):
    funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant=f"integral({quadrature_degree})")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="spectral")

    funspace_CG1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")
    funspace_DG0 = FunctionSpace(mesh, "DG", 0, variant=f"integral({quadrature_degree})")
    mixspace_CG1_Ned1 = MixedFunctionSpace([funspace_CG1, funspace_Ned1])

    (x, y) = SpatialCoordinate(mesh)
    f = Function(funspace_Ned).interpolate(as_vector([sin(pi * (1-x-y)), sin(pi/2 * (1-x-y))]))
    q = Function(funspace_DG).interpolate(curl(f))


    p0, u0 = TrialFunctions(mixspace_CG1_Ned1)
    p1, u1 = TestFunctions(mixspace_CG1_Ned1)
    PIq_mixed = Function(mixspace_CG1_Ned1)

    a = (inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx + inner(p0, p1)*dx - inner(u0, grad(p1))*dx)


    PMq = Function(funspace_Ned)
    B = Constant(bogovskii._map_DG_face(degree, variant="spectral"))

    ndof_Ned = funspace_Ned.finat_element.space_dimension()
    ndof_DG = funspace_DG.finat_element.space_dimension()

    domain = f"{{[i,j]: 0 <= i < {ndof_Ned} and 0 <= j < {ndof_DG}}}"
    instructions = f"""
    for i
        a[i] = 0.0
        for j
            a[i] = a[i] + T[i*{ndof_DG} + j] * b[j]
        end
    end
    """


    time_start = time.time()
    
    n_solves = 0
    while True:
        for i in range(MIN_N_SOLVES):   
            Iq = Function(funspace_DG0).interpolate(q) 
            L = inner(Iq, curl(u1))*dx
            solve(a==L, PIq_mixed)
            _, PIq = PIq_mixed.subfunctions

            Mq = Function(funspace_DG).interpolate(q - Iq)
            Mq_scaled = Function(funspace_DG).interpolate(JacobianDeterminant(mesh) * Mq)
            par_loop(
                (domain, instructions),
                dx,
                {"T": (B, READ), "b": (Mq_scaled, READ), "a": (PMq, WRITE)}
            )

        n_solves += MIN_N_SOLVES
        if time.time() - time_start > MIN_RUNTIME_S:
            break

    time_elapsed = time.time() - time_start
    
    Pq = Function(funspace_Ned).interpolate(PIq + PMq)
    assert norm(curl(Pq) - q, "L2") < 1e-10

    return time_elapsed, n_solves


data_naive = []
data_local = []

data_fields = ["mesh_id", "mesh_size", "polynomial_degree", "number_of_dof", "time_elapsed", "number_of_solves"]

time_start = time.time()
for mesh_level, mesh in enumerate(MeshHierarchy(base_mesh, experiment_n_refinements)):

    print(f"====   Mesh level{mesh_level:3}  ====")

    funspace_DG0 = FunctionSpace(mesh, "DG", 0)
    mesh_size = Function(funspace_DG0).interpolate(CellDiameter(mesh)).dat.data_ro.max()

    data_naive.append({"mesh_id": mesh_level, "mesh_size": mesh_size})
    data_local.append({"mesh_id": mesh_level, "mesh_size": mesh_size})

    for degree in experiment_degrees:
        print(f"{int(time.time() - time_start):4}s elapsed;  degree{degree:3}", end="")

        funspace_DG = FunctionSpace(mesh, "DG", degree-1)
        data_naive[-1]["polynomial_degree"] = \
        data_local[-1]["polynomial_degree"] = degree
        data_naive[-1]["number_of_dof"] = \
        data_local[-1]["number_of_dof"] = funspace_DG.dim()

        gc.collect()
        data_naive[-1]["time_elapsed"], data_naive[-1]["number_of_solves"] = runtime_naive(mesh, degree)
        print(f";  naive{data_naive[-1]["time_elapsed"] / data_naive[-1]["number_of_solves"]:8.4f}", end="")

        gc.collect()
        data_local[-1]["time_elapsed"], data_local[-1]["number_of_solves"] = runtime_local(mesh, degree)
        print(f";  local{data_local[-1]["time_elapsed"] / data_local[-1]["number_of_solves"]:8.4f}")

    print()


with open("output/discrete-poincare/DG-naive.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_naive)

with open("output/discrete-poincare/DG-local.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_local)