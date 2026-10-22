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

experiment_degrees = [[2, 3], [2, 3], [2, 3], [2, 3], [2]]
experiment_n_refinements = 5

quadrature_degree = 4


solver_parameters = {
    "mat_type": "nest",
    "snes_type": "ksponly",

    "ksp_type": "minres",
    "ksp_rtol": 1e-10,
    "ksp_max_it": 1000,

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

def runtime_global(mesh, degree):

    def benchmark():
        funspace_Ned1 = FunctionSpace(mesh, "N1curl", degree)
        funspace_RT = FunctionSpace(mesh, "RT", degree)
        funspace_DG = FunctionSpace(mesh, "DG", degree - 1)
        mixspace_Ned1_RT = MixedFunctionSpace([funspace_Ned1, funspace_RT])

        f = Function(funspace_RT)
        q = Function(funspace_DG)

        Pq_mixed = Function(mixspace_Ned1_RT)
        u0, v0 = TrialFunctions(mixspace_Ned1_RT)
        u1, v1 = TestFunctions(mixspace_Ned1_RT)

        a = (inner(curl(u0), v1)*dx + inner(div(v0), div(v1))*dx - inner(u0, u1)*dx + inner(v0, curl(u1))*dx)
        aP = (inner(u0, u1)*dx + inner(v0, v1)*dx + inner(div(v0), div(v1))*dx)

        f.dat.data[:] = np.random.rand(funspace_RT.dim())
        q.interpolate(div(f))

        L = inner(q, div(v1))*dx
        solve(a==L, Pq_mixed, solver_parameters=solver_parameters, Jp=aP)

        _, Pq = Pq_mixed.subfunctions
        return q, Pq


    # warm up
    q, Pq = benchmark()
    assert norm(div(Pq) - q, "L2") / norm(q, "L2") < 1e-9

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

    def benchmark():
        funspace_RT = FunctionSpace(mesh, "RT", degree, variant=f"integral")
        funspace_DG = FunctionSpace(mesh, "DG", degree - 1, variant=f"spectral")

        funspace_Ned1_1 = FunctionSpace(mesh, "N1curl", 1, variant=f"integral({quadrature_degree})")
        funspace_RT_1 = FunctionSpace(mesh, "RT", 1, variant=f"integral({quadrature_degree})")
        funspace_DG_0 = FunctionSpace(mesh, "DG", 0, variant=f"integral({quadrature_degree})")
        mixspace_Ned1_RT_1 = MixedFunctionSpace([funspace_Ned1_1, funspace_RT_1])

        f = Function(funspace_RT)
        q = Function(funspace_DG)
        
        u0, v0 = TrialFunctions(mixspace_Ned1_RT_1)
        u1, v1 = TestFunctions(mixspace_Ned1_RT_1)
        PIq_mixed = Function(mixspace_Ned1_RT_1)

        a = (inner(curl(u0), v1)*dx + inner(div(v0), div(v1))*dx - inner(u0, u1)*dx + inner(v0, curl(u1))*dx)
        aP = (inner(u0, u1)*dx + inner(v0, v1)*dx + inner(div(v0), div(v1))*dx)
        Iq = Function(funspace_DG_0)

        detJ = JacobianDeterminant(mesh) 
        Mq_scaled = Function(funspace_DG)
        PMq = Function(funspace_RT)
        B = Constant(bogovskii_map(funspace_DG, funspace_RT))

        Pq = Function(funspace_RT)

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


        f.dat.data[:] = np.random.rand(funspace_RT.dim())
        q.interpolate(div(f))
        Iq.interpolate(q)

        L = inner(Iq, div(v1))*dx
        solve(a==L, PIq_mixed, solver_parameters=solver_parameters, Jp=aP)

        Mq_scaled.interpolate(detJ * (q - Iq))
        par_loop(
            (domain, instructions),
            dx,
            {"T": (B, READ), "b": (Mq_scaled, READ), "a": (PMq, WRITE)}
        )

        _, PIq = PIq_mixed.subfunctions
        Pq.interpolate(PIq + PMq)
        return q, Pq


    # warm up
    q, Pq = benchmark()
    assert norm(div(Pq) - q, "L2") / norm(q, "L2") < 1e-9

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

    for degree in experiment_degrees[mesh_level]:
        print(f"{int(perf_counter() - time_start):4}s elapsed;  degree{degree:3}", end="")

        funspace_DG = FunctionSpace(mesh, "DG", degree - 1)
        ndof_DG = funspace_DG.dim()

        data_global.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_DG})
        data_local.append({"mesh_id": mesh_level, "mesh_size": mesh_size, "polynomial_degree": degree, "number_of_dof": ndof_DG})
        
        gc.collect()
        np.random.seed(0)
        data_global[-1]["mean_runtime"], data_global[-1]["median_runtime"], data_global[-1]["minimum_runtime"] = runtime_global(mesh, degree)
        print(f";  global{data_global[-1]["median_runtime"]:8.4f}", end="")

        gc.collect()
        np.random.seed(0)
        data_local[-1]["mean_runtime"], data_local[-1]["median_runtime"], data_local[-1]["minimum_runtime"] = runtime_local(mesh, degree)
        print(f";  local{data_local[-1]["median_runtime"]:8.4f}")

    print()


with open("output/DG-global-setup.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_global)

with open("output/DG-local-setup.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data_fields)
    writer.writeheader()
    writer.writerows(data_local)