import numpy as np
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
experiment_n_refinements = 3

quadrature_degree = 4


solver_parameters = {
    "mat_type": "nest",
    "snes_type": "ksponly",

    "ksp_type": "minres",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 100,
    "ksp_converged_reason": None,

    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",

    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "jacobi",
    },

    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
}

solver_parameters = {
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 100,
    "ksp_converged_reason": None,

    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",

    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "jacobi",
    },

    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
}


def runtime_global(mesh, degree):
    funspace_CG = FunctionSpace(mesh, "CG", degree)
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree)
    funspace_RT = FunctionSpace(mesh, "RT", degree)
    mixspace_CG_Ned1 = MixedFunctionSpace([funspace_CG, funspace_Ned1])

    f = Function(funspace_Ned1)
    v = Function(funspace_RT)

    Pv_mixed = Function(mixspace_CG_Ned1)
    p0, u0 = TrialFunctions(mixspace_CG_Ned1)
    p1, u1 = TestFunctions(mixspace_CG_Ned1)

    a = (inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx + inner(p0, p1)*dx + inner(u0, grad(p1))*dx)

    aP = (
        inner(p0, p1)*dx
        + inner(u0, u1)*dx
        + inner(curl(u0), curl(u1))*dx
    )

    L = inner(v, curl(u1))*dx
    problem = LinearVariationalProblem(a, L, Pv_mixed, aP=aP, constant_jacobian=True)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)

    def benchmark():
        f.dat.data[:] = np.random.rand(funspace_Ned1.dim())
        v.interpolate(curl(f))

        Pv_mixed.assign(0)
        solver.solve()

        _, Pv = Pv_mixed.subfunctions
        return v, Pv


    # warm up
    v, Pv = benchmark()
    print("\n", norm(curl(Pv) - v, "L2") / norm(v, "L2"))
    assert norm(curl(Pv) - v, "L2") / norm(v, "L2") < 1e-4

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



def runtime_local(mesh, degree):
    funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree, variant=f"integral")
    funspace_RT = FunctionSpace(mesh, "RT", degree, variant=f"integral")

    funspace_CG_1 = FunctionSpace(mesh, "CG", 1, variant="equispaced")
    funspace_Ned1_1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")
    funspace_RT_1 = FunctionSpace(mesh, "RT", 1, variant=f"integral({quadrature_degree})")
    mixspace_CG_Ned1_1 = MixedFunctionSpace([funspace_CG_1, funspace_Ned1_1])

    f = Function(funspace_Ned1)
    v = Function(funspace_RT)
    
    p0, u0 = TrialFunctions(mixspace_CG_Ned1_1)
    p1, u1 = TestFunctions(mixspace_CG_Ned1_1)
    PIv_mixed = Function(mixspace_CG_Ned1_1)

    a = (inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx + inner(p0, p1)*dx + inner(u0, grad(p1))*dx)
    Iv = Function(funspace_RT_1)

    aP = (
        inner(p0, p1)*dx
        + inner(u0, u1)*dx
        + inner(curl(u0), curl(u1))*dx
    )

    L = inner(Iv, curl(u1))*dx
    problem = LinearVariationalProblem(a, L, PIv_mixed, aP=aP, constant_jacobian=True)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)

    Mv = Function(funspace_RT)
    PMv = Function(funspace_Ned1)
    B = Constant(bogovskii_map(funspace_RT, funspace_Ned1))

    Pv = Function(funspace_Ned1)

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

    def benchmark():
        f.dat.data[:] = np.random.rand(funspace_Ned1.dim())
        v.interpolate(curl(f))
        Iv.interpolate(v)

        PIv_mixed.assign(0)
        solver.solve()

        Mv.interpolate(v - Iv)
        par_loop(
            (domain, instructions),
            dx,
            {"T": (B, READ), "b": (Mv, READ), "a": (PMv, WRITE)}
        )

        _, PIv = PIv_mixed.subfunctions
        Pv.interpolate(PIv + PMv)
        return v, Pv


    # warm up
    v, Pv = benchmark()
    assert norm(curl(Pv) - v, "L2") / norm(v, "L2") < 1e-4

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
for mesh_level in range(3, 4):

    print(f"====   Mesh level{mesh_level:3}  ====")

    shape = Box(Pnt(0.0, 0.0, 0.0), Pnt(1.0, 1.0, 1.0))
    geo = OCCGeometry(shape)
    ngmesh = geo.GenerateMesh(maxh=1.0/(2**mesh_level))
    mesh = Mesh(ngmesh)

    funspace_DG_0 = FunctionSpace(mesh, "DG", 0)
    mesh_size = Function(funspace_DG_0).interpolate(CellDiameter(mesh)).dat.data_ro.max()

    for degree in experiment_degrees:
        print(f"{int(perf_counter() - time_start):4}s elapsed;  degree{degree:3}", end="")

        funspace_RT = FunctionSpace(mesh, "RT", degree)
        ndof_RT = funspace_RT.dim()

        data_global.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_RT})
        data_local.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_RT})

        gc.collect()
        np.random.seed(0)
        data_global[-1]["mean_runtime"], data_global[-1]["median_runtime"], data_global[-1]["minimum_runtime"] = runtime_global(mesh, degree)
        print(f";  global{data_global[-1]["median_runtime"]:8.4f}", end="")

        gc.collect()
        np.random.seed(0)
        data_local[-1]["mean_runtime"], data_local[-1]["median_runtime"], data_local[-1]["minimum_runtime"] = runtime_local(mesh, degree)
        print(f";  local{data_local[-1]["median_runtime"]:8.4f}")

    print()


with open("output/RT-global-repeated.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_global)

with open("output/RT-local-repeated.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_local)