import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
import csv, time
import gc
import bogovskii

MIN_N_SOLVES = 30
MIN_RUNTIME_S = 10

experiment_degrees = [2, 3, 4]
experiment_n_refinements = 7


base_mesh = UnitTriangleMesh(0)
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


    Pu = Function(funspace_CG)
    p0 = TrialFunction(funspace_CG)
    p1 = TestFunction(funspace_CG)

    a = inner(grad(p0), grad(p1))*dx
    nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)


    time_start = time.time()
    
    n_solves = 0
    while True:
        for i in range(MIN_N_SOLVES):
            L = inner(u, grad(p1))*dx
            solve(a==L, Pu, nullspace=nullspace)
        
        n_solves += MIN_N_SOLVES
        if time.time() - time_start > MIN_RUNTIME_S:
            break

    time_elapsed = time.time() - time_start

    assert norm(grad(Pu) - u, "L2") < 1e-10

    return time_elapsed, n_solves


def runtime_local(mesh, degree):
    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant=f"integral({quadrature_degree})")

    funspace_CG1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")

    (x, y) = SpatialCoordinate(mesh)
    f = Function(funspace_CG).interpolate(sin(pi * (1-x-y)))
    u = Function(funspace_Ned).interpolate(grad(f))


    PIu = Function(funspace_CG1)
    p0 = TrialFunction(funspace_CG1)
    p1 = TestFunction(funspace_CG1)

    a = inner(grad(p0), grad(p1))*dx
    nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)


    PMu = Function(funspace_CG)
    B = Constant(bogovskii.map(degree, "face"))

    ndof_CG = funspace_CG.finat_element.space_dimension()
    ndof_Ned = funspace_Ned.finat_element.space_dimension()

    domain = f"{{[i,j]: 0 <= i < {ndof_CG} and 0 <= j < {ndof_Ned}}}"
    instructions = f"""
    for i
        a[i] = 0.0
        for j
            a[i] = a[i] + T[i*{ndof_Ned} + j] * b[j]
        end
    end
    """


    time_start = time.time()
    
    n_solves = 0
    while True:
        for i in range(MIN_N_SOLVES):   
            Iu = Function(funspace_Ned1).interpolate(u)
            L = inner(Iu, grad(p1))*dx
            solve(a==L, PIu, nullspace=nullspace)

            Mu = Function(funspace_Ned).interpolate(u - Iu)
            par_loop(
                (domain, instructions),
                dx,
                {"T": (B, READ), "b": (Mu, READ), "a": (PMu, WRITE)}
            )

        n_solves += MIN_N_SOLVES
        if time.time() - time_start > MIN_RUNTIME_S:
            break

    time_elapsed = time.time() - time_start
    
    Pu = Function(funspace_CG).interpolate(PIu + PMu)
    assert norm(grad(Pu) - u, "L2") < 1e-10

    return time_elapsed, n_solves


data_naive = []
data_local = []

data_fields = ["mesh_id", "mesh_size", "polynomial_degree", "number_of_dof", "time_elapsed", "number_of_solves"]

time_start = time.time()
for mesh_level, mesh in enumerate(MeshHierarchy(base_mesh, experiment_n_refinements)):

    if mesh_level == 0:
        continue

    print(f"====   Mesh level{mesh_level:3}  ====")

    funspace_DG0 = FunctionSpace(mesh, "DG", 0)
    mesh_size = Function(funspace_DG0).interpolate(CellDiameter(mesh)).dat.data_ro.max()

    for degree in experiment_degrees:
        print(f"{int(time.time() - time_start):4}s elapsed;  degree{degree:3}", end="")

        funspace_Ned = FunctionSpace(mesh, "N1curl", degree)
        ndof_Ned = funspace_Ned.dim()

        data_naive.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_Ned})
        data_local.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_Ned})

        gc.collect()
        data_naive[-1]["time_elapsed"], data_naive[-1]["number_of_solves"] = runtime_naive(mesh, degree)
        print(f";  naive{data_naive[-1]["time_elapsed"] / data_naive[-1]["number_of_solves"]:8.4f}", end="")

        gc.collect()
        data_local[-1]["time_elapsed"], data_local[-1]["number_of_solves"] = runtime_local(mesh, degree)
        print(f";  local{data_local[-1]["time_elapsed"] / data_local[-1]["number_of_solves"]:8.4f}")

    print()


with open("output/discrete-poincare/Ned-naive.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_naive)

with open("output/discrete-poincare/Ned-local.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_local)