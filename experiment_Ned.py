import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
import bogovskii
import csv, time


MIN_N_RUNS = 50
MIN_RUNTIME_S = 10

experiment_degrees = [2, 4, 6, 8]
experiment_n_refinements = 2


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
    funspace_CG = FunctionSpace(mesh, "CG", degree)
    funspace_Ned = FunctionSpace(mesh, "N1curl", degree)

    (x, y) = SpatialCoordinate(mesh)
    f = Function(funspace_CG).interpolate(sin(pi * (1-x-y)))
    u = Function(funspace_Ned).interpolate(grad(f))


    Pu_data = Function(funspace_CG)
    Pu = TrialFunction(funspace_CG)
    p = TestFunction(funspace_CG)

    a = inner(grad(Pu), grad(p))*dx
    nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)


    time_start = time.time()
    
    n_runs = 0
    while time.time() - time_start < MIN_RUNTIME_S:
        for i in range(MIN_N_RUNS):
            L = inner(u, grad(p))*dx
            solve(a==L, Pu_data, nullspace=nullspace)
        n_runs += MIN_N_RUNS

    time_elapsed = time.time() - time_start

    assert norm(grad(Pu_data) - u, "L2") < 1e-10

    return time_elapsed / n_runs


def runtime_local(mesh, degree):
    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant=f"integral({quadrature_degree})")

    funspace_CG1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")

    (x, y) = SpatialCoordinate(mesh)
    f = Function(funspace_CG).interpolate(sin(pi * (1-x-y)))
    u = Function(funspace_Ned).interpolate(grad(f))


    PIu_data = Function(funspace_CG1)
    PIu = TrialFunction(funspace_CG1)
    p = TestFunction(funspace_CG1)

    a = inner(grad(PIu), grad(p))*dx
    nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)


    PMu = Function(funspace_CG)
    B = Constant(bogovskii.map(degree, "face"))

    ndof_CG = funspace_CG.finat_element.space_dimension()
    ndof_Ned = funspace_Ned.finat_element.space_dimension()

    domain = f"{{[i,j]: 0 <= i < {ndof_CG} and 0 <= j < {ndof_Ned}}}"
    instructions = f"""
    for i
    PMu[i] = 0.0
    for j
        PMu[i] = PMu[i] + B[i*{ndof_Ned} + j] * Mu[j]
    end
    end
    """


    time_start = time.time()
    
    n_runs = 0
    while time.time() - time_start < MIN_RUNTIME_S:
        for i in range(MIN_N_RUNS):   
            Iu = Function(funspace_Ned1).interpolate(u)
            L = inner(Iu, grad(p))*dx
            solve(a==L, PIu_data, nullspace=nullspace)

            Mu = Function(funspace_Ned).interpolate(u - Iu)
            par_loop(
                (domain, instructions),
                dx,
                {"Mu": (Mu, READ), "PMu": (PMu, WRITE), "B": (B, READ)}
            )

        n_runs += MIN_N_RUNS

    time_elapsed = time.time() - time_start
    
    Pu = Function(funspace_CG).interpolate(PIu_data + PMu)
    assert norm(grad(Pu) - u, "L2") < 1e-10

    return time_elapsed / n_runs


data_naive = [experiment_degrees]
data_local = [experiment_degrees]

time_start = time.time()
for mesh_level, mesh in enumerate(MeshHierarchy(base_mesh, experiment_n_refinements)):

    print(f"====   Mesh level{mesh_level:3}  ====")

    data_naive.append([])
    data_local.append([])

    for degree in experiment_degrees:
        print(f"{int(time.time() - time_start):4}s elapsed;  degree{degree:3}", end="")

        rt_naive = runtime_naive(mesh, degree)
        print(f";  naive{rt_naive:8.4f}", end="")

        rt_local = runtime_local(mesh, degree)
        print(f";  local{rt_local:8.4f}")

        data_naive[-1].append(rt_naive)
        data_local[-1].append(rt_local)

    print()


with open("output/discrete-poincare/Ned-naive.csv", "w+", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_naive)

with open("output/discrete-poincare/Ned-local.csv", "w+", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_local)