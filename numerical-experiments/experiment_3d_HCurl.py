import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
import csv, time
import gc
from bogovskii import bogovskii_map

MIN_N_SOLVES = 2
MIN_RUNTIME_S = 1

experiment_degrees = [2, 3, 4]
experiment_n_refinements = 2

quadrature_degree = 6


def runtime_global(mesh, degree):
    funspace_CG = FunctionSpace(mesh, "CG", degree)
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree)

    f = Function(funspace_CG)

    Pu = Function(funspace_CG)
    p0 = TrialFunction(funspace_CG)
    p1 = TestFunction(funspace_CG)

    a = inner(grad(p0), grad(p1))*dx
    nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)


    time_start = time.time()
    
    n_solves = 0
    while True:
        for i in range(MIN_N_SOLVES):
            f.dat.data[:] = np.random.rand(funspace_CG.dim())
            u = Function(funspace_Ned1).interpolate(grad(f))

            L = inner(u, grad(p1))*dx
            solve(a==L, Pu, nullspace=nullspace)

            assert norm(grad(Pu) - u, "L2") < 1e-10
        
        n_solves += MIN_N_SOLVES
        if time.time() - time_start > MIN_RUNTIME_S:
            break

    time_elapsed = time.time() - time_start
    return time_elapsed, n_solves


def runtime_local(mesh, degree):
    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree, variant=f"integral({quadrature_degree})")

    funspace_CG_1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
    funspace_Ned1_1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")

    f = Function(funspace_CG)
    
    PIu = Function(funspace_CG_1)
    p0 = TrialFunction(funspace_CG_1)
    p1 = TestFunction(funspace_CG_1)

    a = inner(grad(p0), grad(p1))*dx
    nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)


    PMu = Function(funspace_CG)
    B = Constant(bogovskii_map(funspace_Ned1, funspace_CG))

    ndof_CG = funspace_CG.finat_element.space_dimension()
    ndof_Ned1 = funspace_Ned1.finat_element.space_dimension()

    domain = f"{{[i,j]: 0 <= i < {ndof_CG} and 0 <= j < {ndof_Ned1}}}"
    instructions = f"""
    for i
        a[i] = 0.0
        for j
            a[i] = a[i] + T[i*{ndof_Ned1} + j] * b[j]
        end
    end
    """


    time_start = time.time()
    
    n_solves = 0
    while True:
        for i in range(MIN_N_SOLVES):   
            f.dat.data[:] = np.random.rand(funspace_CG.dim())
            u = Function(funspace_Ned1).interpolate(grad(f))

            Iu = Function(funspace_Ned1_1).interpolate(u)
            L = inner(Iu, grad(p1))*dx
            solve(a==L, PIu, nullspace=nullspace)

            Mu = Function(funspace_Ned1).interpolate(u - Iu)
            par_loop(
                (domain, instructions),
                dx,
                {"T": (B, READ), "b": (Mu, READ), "a": (PMu, WRITE)}
            )

            Pu = Function(funspace_CG).interpolate(PIu + PMu)
            assert norm(grad(Pu) - u, "L2") < 1e-10

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

        funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree)
        ndof_Ned1 = funspace_Ned1.dim()

        data_global.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_Ned1})
        data_local.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_Ned1})

        gc.collect()
        np.random.seed(0)
        data_global[-1]["time_elapsed"], data_global[-1]["number_of_solves"] = runtime_global(mesh, degree)
        print(f";  global{data_global[-1]["time_elapsed"] / data_global[-1]["number_of_solves"]:8.4f}", end="")

        gc.collect()
        np.random.seed(0)
        data_local[-1]["time_elapsed"], data_local[-1]["number_of_solves"] = runtime_local(mesh, degree)
        print(f";  local{data_local[-1]["time_elapsed"] / data_local[-1]["number_of_solves"]:8.4f}")

    print()


with open("output/discrete-poincare/Ned-global.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_global)

with open("output/discrete-poincare/Ned-local.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_local)