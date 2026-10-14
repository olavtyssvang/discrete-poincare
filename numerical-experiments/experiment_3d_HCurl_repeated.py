import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from netgen.occ import *
from time import perf_counter
import csv
import gc
from bogovskii import bogovskii_map

MIN_BATCHTIME_S = 0.1 
MIN_N_BATCHES = 9
MIN_RUNTIME_S = 1.0

experiment_degrees = [2, 3]
experiment_n_refinements = 4

quadrature_degree = 4


solver_parameters = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "ksp_rtol": 1e-10,
    "ksp_atol": 1e-12,
    "ksp_max_it": 1000,
}

def runtime_global(mesh, degree):
    funspace_CG = FunctionSpace(mesh, "CG", degree)
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree)

    f = Function(funspace_CG)
    u = Function(funspace_Ned1)

    Pu = Function(funspace_CG)
    p0 = TrialFunction(funspace_CG)
    p1 = TestFunction(funspace_CG)

    a = inner(grad(p0), grad(p1))*dx
    nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)

    L = inner(u, grad(p1))*dx
    problem = LinearVariationalProblem(a, L, Pu)
    solver = LinearVariationalSolver(problem, nullspace=nullspace, solver_parameters=solver_parameters)

    def benchmark():
        f.dat.data[:] = np.random.randn(funspace_CG.dim())
        u.interpolate(grad(f))

        Pu.assign(0)
        solver.solve()

        return u, Pu


    # warm up
    u, Pu = benchmark()
    assert norm(grad(Pu) - u, "L2") / norm(u, "L2") < 1e-8

    # run benchmarks
    n_solves = 0
    n_batches = 0
    runtime = []

    time_start = perf_counter()
    time_batch = time_start
    
    n_solves = 0
    while True:
        benchmark()
        n_solves += 1

        # loop batch
        time_now = perf_counter()
        if time_now - time_batch < MIN_BATCHTIME_S:
            continue

        # record runtime
        runtime.append((time_now - time_batch) / n_solves)
        n_batches += 1

        # repeat if need be
        if time_now - time_start < MIN_RUNTIME_S or n_batches < MIN_N_BATCHES:
            time_batch = time_now
            n_solves = 0
            continue
        else:
            break

    return np.mean(runtime), np.median(runtime), np.min(runtime)


def runtime_local(mesh, degree):
    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree, variant=f"integral")

    funspace_CG_1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
    funspace_Ned1_1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")

    f = Function(funspace_CG)
    u = Function(funspace_Ned1)

    p0 = TrialFunction(funspace_CG_1)
    p1 = TestFunction(funspace_CG_1)
    PIu = Function(funspace_CG_1)

    a = inner(grad(p0), grad(p1))*dx
    nullspace = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
    Iu = Function(funspace_Ned1_1)

    L = inner(Iu, grad(p1))*dx
    problem = LinearVariationalProblem(a, L, PIu)
    solver = LinearVariationalSolver(problem, nullspace=nullspace, solver_parameters=solver_parameters)

    Mu = Function(funspace_Ned1)
    PMu = Function(funspace_CG)
    B = Constant(bogovskii_map(funspace_Ned1, funspace_CG))

    Pu = Function(funspace_CG)

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

    def benchmark():
        f.dat.data[:] = np.random.randn(funspace_CG.dim())
        u.interpolate(grad(f))
        Iu.interpolate(u)

        PIu.assign(0)
        solver.solve()

        Mu.interpolate(u - Iu)
        par_loop(
            (domain, instructions),
            dx,
            {"T": (B, READ), "b": (Mu, READ), "a": (PMu, WRITE)}
        )

        Pu.interpolate(PIu + PMu)
        return u, Pu

    
    # warm up
    u, Pu = benchmark()
    assert norm(grad(Pu) - u, "L2") / norm(u, "L2") < 1e-8

    # run benchmarks
    n_solves = 0
    n_batches = 0
    runtime = []

    time_start = perf_counter()
    time_batch = time_start

    while True:
        benchmark()
        n_solves += 1

        # loop batch
        time_now = perf_counter()
        if time_now - time_batch < MIN_BATCHTIME_S:
            continue

        # record runtime
        runtime.append((time_now - time_batch) / n_solves)
        n_batches += 1

        # repeat if need be
        if time_now - time_start < MIN_RUNTIME_S or n_batches < MIN_N_BATCHES:
            time_batch = time_now
            n_solves = 0
            continue
        else:
            break

    return np.mean(runtime), np.median(runtime), np.min(runtime)


data_global = []
data_local = []

data_fields = ["mesh_id", "mesh_size", "polynomial_degree", "number_of_dof", "mean_runtime", "median_runtime", "minimum_runtime"]

time_start = perf_counter()
for mesh_level in range(experiment_n_refinements):

    print(f"====   Mesh level{mesh_level:3}  ====")

    shape = Box(Pnt(0.0, 0.0, 0.0), Pnt(1.0, 1.0, 1.0))
    geo = OCCGeometry(shape)
    ngmesh = geo.GenerateMesh(maxh=1.0/(2**mesh_level))
    mesh = Mesh(ngmesh)

    funspace_DG_0 = FunctionSpace(mesh, "DG", 0)
    mesh_size = Function(funspace_DG_0).interpolate(CellDiameter(mesh)).dat.data_ro.max()

    for degree in experiment_degrees:
        print(f"{int(perf_counter() - time_start):4}s elapsed;  degree{degree:3}", end="")

        funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree)
        ndof_Ned1 = funspace_Ned1.dim()

        data_global.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_Ned1})
        data_local.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_Ned1})

        gc.collect()
        np.random.seed(0)
        data_global[-1]["mean_runtime"], data_global[-1]["median_runtime"], data_global[-1]["minimum_runtime"] = runtime_global(mesh, degree)
        print(f";  global{data_global[-1]["median_runtime"]:8.4f}", end="")

        gc.collect()
        np.random.seed(0)
        data_local[-1]["mean_runtime"], data_local[-1]["median_runtime"], data_local[-1]["minimum_runtime"] = runtime_local(mesh, degree)
        print(f";  local{data_local[-1]["median_runtime"]:8.4f}")

    print()


with open("output/Ned1-global-repeated.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_global)

with open("output/Ned1-local-repeated.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_local)