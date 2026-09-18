import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
import csv, time
import gc
from bogovskii import bogovskii_map

MIN_N_SOLVES = 5
MIN_RUNTIME_S = 10

experiment_degrees = [2, 3]
experiment_n_refinements = 4

quadrature_degree = 6


def runtime_global(mesh, degree):
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree)
    funspace_RT = FunctionSpace(mesh, "RT", degree)
    funspace_DG = FunctionSpace(mesh, "DG", degree - 1)
    mixspace_Ned1_RT = MixedFunctionSpace([funspace_Ned1, funspace_RT])

    f = Function(funspace_RT)

    Pq_mixed = Function(mixspace_Ned1_RT)
    u0, v0 = TrialFunctions(mixspace_Ned1_RT)
    u1, v1 = TestFunctions(mixspace_Ned1_RT)

    a = (inner(curl(u0), v1)*dx + inner(div(v0), div(v1))*dx + inner(u0, u1)*dx - inner(v0, curl(u1))*dx)


    time_start = time.time()
    
    n_solves = 0
    while True:
        for i in range(MIN_N_SOLVES):
            f.dat.data[:] = np.random.rand(funspace_RT.dim())
            q = Function(funspace_DG).interpolate(div(f))

            L = inner(q, div(v1))*dx
            solve(a==L, Pq_mixed)

            _, Pq = Pq_mixed.subfunctions
            assert norm(div(Pq) - q, "L2") < 1e-8
        
        n_solves += MIN_N_SOLVES
        if time.time() - time_start > MIN_RUNTIME_S:
            break

    time_elapsed = time.time() - time_start
    return time_elapsed, n_solves


def runtime_local(mesh, degree):
    funspace_RT = FunctionSpace(mesh, "RT", degree, variant=f"integral({quadrature_degree})")
    funspace_DG = FunctionSpace(mesh, "DG", degree - 1, variant=f"spectral")

    funspace_Ned1_1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")
    funspace_RT_1 = FunctionSpace(mesh, "RT", 1, variant=f"integral({quadrature_degree})")
    funspace_DG_0 = FunctionSpace(mesh, "DG", 0, variant=f"integral({quadrature_degree})")
    mixspace_Ned1_RT_1 = MixedFunctionSpace([funspace_Ned1_1, funspace_RT_1])

    f = Function(funspace_RT)
    
    u0, v0 = TrialFunctions(mixspace_Ned1_RT_1)
    u1, v1 = TestFunctions(mixspace_Ned1_RT_1)
    PIq_mixed = Function(mixspace_Ned1_RT_1)

    a = (inner(curl(u0), v1)*dx + inner(div(v0), div(v1))*dx + inner(u0, u1)*dx - inner(v0, curl(u1))*dx)
    

    PMq = Function(funspace_RT)
    B = Constant(bogovskii_map(funspace_DG, funspace_RT))

    ndof_RT = funspace_RT.finat_element.space_dimension()
    ndof_DG = funspace_DG.finat_element.space_dimension()

    domain = f"{{[i,j]: 0 <= i < {ndof_RT} and 0 <= j < {ndof_DG}}}"
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
            f.dat.data[:] = np.random.rand(funspace_RT.dim())
            q = Function(funspace_DG).interpolate(div(f))

            Iq = Function(funspace_DG_0).interpolate(q)
            L = inner(Iq, div(v1))*dx
            solve(a==L, PIq_mixed)

            Mq = Function(funspace_DG).interpolate(q - Iq)
            Mq_scaled = Function(funspace_DG).interpolate(JacobianDeterminant(mesh) * Mq)
            par_loop(
                (domain, instructions),
                dx,
                {"T": (B, READ), "b": (Mq_scaled, READ), "a": (PMq, WRITE)}
            )

            _, PIq = PIq_mixed.subfunctions
            Pq = Function(funspace_RT).interpolate(PIq + PMq)
            assert norm(div(Pq) - q, "L2") < 1e-8

        n_solves += MIN_N_SOLVES
        if time.time() - time_start > MIN_RUNTIME_S:
            break

    time_elapsed = time.time() - time_start

    return time_elapsed, n_solves


data_global = []
data_local = []

data_fields = ["mesh_id", "mesh_size", "polynomial_degree", "number_of_dof", "time_elapsed", "number_of_solves"]

time_start = time.time()
for mesh_level in range(experiment_n_refinements):

    print(f"====   Mesh level{mesh_level:3}  ====")

    nx = ny = nz = 2 ** mesh_level
    mesh = UnitCubeMesh(nx, ny, nz)

    funspace_DG_0 = FunctionSpace(mesh, "DG", 0)
    mesh_size = Function(funspace_DG_0).interpolate(CellDiameter(mesh)).dat.data_ro.max()

    for degree in experiment_degrees:
        print(f"{int(time.time() - time_start):4}s elapsed;  degree{degree:3}", end="")

        funspace_DG = FunctionSpace(mesh, "DG", degree)
        ndof_DG = funspace_DG.dim()

        data_global.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_DG})
        data_local.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_DG})

        gc.collect()
        np.random.seed(0)
        data_global[-1]["time_elapsed"], data_global[-1]["number_of_solves"] = runtime_global(mesh, degree)
        print(f";  global{data_global[-1]["time_elapsed"] / data_global[-1]["number_of_solves"]:8.4f}", end="")

        gc.collect()
        np.random.seed(0)
        data_local[-1]["time_elapsed"], data_local[-1]["number_of_solves"] = runtime_local(mesh, degree)
        print(f";  local{data_local[-1]["time_elapsed"] / data_local[-1]["number_of_solves"]:8.4f}")

    print()


with open("output/discrete-poincare/DG-global.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_global)

with open("output/discrete-poincare/DG-local.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_local)