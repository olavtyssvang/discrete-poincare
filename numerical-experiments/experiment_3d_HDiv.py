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
    funspace_CG = FunctionSpace(mesh, "CG", degree)
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree)
    funspace_RT = FunctionSpace(mesh, "RT", degree)
    mixspace_CG_Ned1 = MixedFunctionSpace([funspace_CG, funspace_Ned1])

    f = Function(funspace_Ned1)

    Pv_mixed = Function(mixspace_CG_Ned1)
    p0, u0 = TrialFunctions(mixspace_CG_Ned1)
    p1, u1 = TestFunctions(mixspace_CG_Ned1)

    a = (inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx + inner(p0, p1)*dx - inner(u0, grad(p1))*dx)


    time_start = time.time()
    
    n_solves = 0
    while True:
        for i in range(MIN_N_SOLVES):
            f.dat.data[:] = np.random.rand(funspace_Ned1.dim())
            v = Function(funspace_RT).interpolate(curl(f))

            L = inner(v, curl(u1))*dx
            solve(a==L, Pv_mixed)

            _, Pv = Pv_mixed.subfunctions
            assert norm(curl(Pv) - v, "L2") < 1e-8
        
        n_solves += MIN_N_SOLVES
        if time.time() - time_start > MIN_RUNTIME_S:
            break

    time_elapsed = time.time() - time_start
    return time_elapsed, n_solves


def runtime_local(mesh, degree):
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree, variant=f"integral({quadrature_degree})")
    funspace_RT = FunctionSpace(mesh, "RT", degree, variant=f"integral({quadrature_degree})")

    funspace_CG_1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
    funspace_Ned1_1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")
    funspace_RT_1 = FunctionSpace(mesh, "RT", 1, variant=f"integral({quadrature_degree})")
    mixspace_CG_Ned1_1 = MixedFunctionSpace([funspace_CG_1, funspace_Ned1_1])

    f = Function(funspace_Ned1)
    
    p0, u0 = TrialFunctions(mixspace_CG_Ned1_1)
    p1, u1 = TestFunctions(mixspace_CG_Ned1_1)
    PIv_mixed = Function(mixspace_CG_Ned1_1)

    a = (inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx + inner(p0, p1)*dx - inner(u0, grad(p1))*dx)


    PMv = Function(funspace_Ned1)
    B = Constant(bogovskii_map(funspace_RT, funspace_Ned1))

    ndof_Ned1 = funspace_Ned1.finat_element.space_dimension()
    ndof_RT = funspace_RT.finat_element.space_dimension()

    domain = f"{{[i,j]: 0 <= i < {ndof_Ned1} and 0 <= j < {ndof_RT}}}"
    instructions = f"""
    for i
        a[i] = 0.0
        for j
            a[i] = a[i] + T[i*{ndof_RT} + j] * b[j]
        end
    end
    """


    time_start = time.time()
    
    n_solves = 0
    while True:
        for i in range(MIN_N_SOLVES):   
            f.dat.data[:] = np.random.rand(funspace_Ned1.dim())
            v = Function(funspace_RT).interpolate(curl(f))

            Iv = Function(funspace_RT_1).interpolate(v)
            L = inner(Iv, curl(u1))*dx
            solve(a==L, PIv_mixed)

            Mv = Function(funspace_RT).interpolate(v - Iv)
            par_loop(
                (domain, instructions),
                dx,
                {"T": (B, READ), "b": (Mv, READ), "a": (PMv, WRITE)}
            )

            _, PIv = PIv_mixed.subfunctions
            Pv = Function(funspace_Ned1).interpolate(PIv + PMv)
            assert norm(curl(Pv) - v, "L2") < 1e-8

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

        funspace_RT = FunctionSpace(mesh, "RT", degree)
        ndof_RT = funspace_RT.dim()

        data_global.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_RT})
        data_local.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_RT})

        gc.collect()
        np.random.seed(0)
        data_global[-1]["time_elapsed"], data_global[-1]["number_of_solves"] = runtime_global(mesh, degree)
        print(f";  global{data_global[-1]["time_elapsed"] / data_global[-1]["number_of_solves"]:8.4f}", end="")

        gc.collect()
        np.random.seed(0)
        data_local[-1]["time_elapsed"], data_local[-1]["number_of_solves"] = runtime_local(mesh, degree)
        print(f";  local{data_local[-1]["time_elapsed"] / data_local[-1]["number_of_solves"]:8.4f}")

    print()


with open("output/discrete-poincare/RT-global.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_global)

with open("output/discrete-poincare/RT-local.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_local)