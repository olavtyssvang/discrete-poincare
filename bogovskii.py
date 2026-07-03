import numpy as np
import scipy.sparse as sp
from firedrake import *


_BOGOVSKII_DEBUG_ENABLE = False

def map(degree, domain):
    if domain == "edge":
        return _map_edge(degree)
    
    if domain == "face":
        return _map_face(degree)
    
    if domain == "tetrahedron":
        return _map_tetrahedron(degree)
    
    raise ValueError(domain)


def npmat(bilinear_form):
    petscmat = assemble(bilinear_form).petscmat
    indptr, indices, data = petscmat.getValuesCSR()
    return sp.csr_array((data, indices, indptr), shape=petscmat.getSize()).toarray()

def ndof_(funspace):
    fe = funspace.finat_element
    dof_map = fe.entity_dofs()
    
    i = 0
    n = len(dof_map)
    out = []

    while i < n and len(dof_map[i][0]) == 0:
        i += 1
    
    while i < n:
        out.append(len(dof_map[i][0]))
        i += 1

    return fe.space_dimension(), *reversed(out)

def _map_edge(degree):
    mesh = UnitIntervalMesh(1)

    funspace_RR = FunctionSpace(mesh, "R", 0)
    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_CG = funspace_CG.finat_element.space_dimension()
    ndof_DG = funspace_DG.finat_element.space_dimension()

    ref_idx_CG = funspace_CG.cell_node_map().values[0]
    ref_idx_DG = funspace_DG.cell_node_map().values[0]

    if _BOGOVSKII_DEBUG_ENABLE:
        out = "==== CG(edge):  ====\n"

        p = Function(funspace_CG)
        for i in range(ndof_CG):
            dof = np.zeros(shape=(ndof_CG))
            dof[i] = 1
            p.dat.data[ref_idx_CG] = dof
            
            for x in [0, 1]:
                out += f"{p(x)[0]: 2.1f}  "
            out += "\n"

        out += "\n==== DG(edge):  ====\n"

        p = Function(funspace_DG)
        for i in range(ndof_DG):
            dof = np.zeros(shape=(ndof_DG))
            dof[i] = 1
            p.dat.data[ref_idx_DG] = dof

            for x in [0, 1]:
                out += f"{p(x)[0]: 2.1f}  "
            out += "\n"

        print(out)

    p0 = TrialFunction(funspace_CG)
    p1 = TestFunction(funspace_CG)

    u0 = TrialFunction(funspace_DG)

    A = npmat(inner(grad(p0), grad(p1))*dx)[np.ix_(ref_idx_CG, ref_idx_CG)]
    M = npmat(inner(u0, p1.dx(0))*dx)[np.ix_(ref_idx_CG, ref_idx_DG)]

    B_edge = np.zeros(shape=(ndof_CG, ndof_DG))
    B_edge[2:, 1:] = np.linalg.inv(A[2:, 2:]) @ M[2:, 1:]

    return B_edge


def _map_face(degree):
    mesh = UnitTriangleMesh(0)
#    (u, v) = SpatialCoordinate(mesh)
#
#    x = u + 1/2 * v
#    y = sqrt(3)/2 * v 
#
#    V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=2)
#    coords = Function(V_coord)
#    coords.interpolate(as_vector([x, y]))
#
#    mesh = Mesh(coords)


    funspace_RR = FunctionSpace(mesh, "R", 0)
    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_CG, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_Ned, ndof_Ned_face, ndof_Ned_edge = ndof_(funspace_Ned)

    ref_idx_CG = funspace_CG.cell_node_map().values[0]
    ref_idx_Ned = funspace_Ned.cell_node_map().values[0]
    ref_idx_DG = funspace_DG.cell_node_map().values[0]

    if _BOGOVSKII_DEBUG_ENABLE:
        out = "==== CG(face):  ====\n"

        p = Function(funspace_CG)
        for i in range(ndof_CG):
            dof = np.zeros(shape=(ndof_CG))
            dof[i] = 1
            p.dat.data[ref_idx_CG] = dof
            
            for x in [(0, 0), (1, 0), (0, 1), (.5, 0), (0, .5), (.5, .5)]:
                out += f"{p(x): 2.1f}  "
            out += "\n"

        out += "\n==== Ned(face): ====\n"

        p = Function(funspace_Ned)
        for i in range(ndof_Ned):
            dof = np.zeros(shape=(ndof_Ned))
            dof[i] = 1
            p.dat.data[ref_idx_Ned] = dof

            for x in [(0, 0), (1, 0), (0, 1)]:
                out += f"{p(x)[0]: 2.1f},{p(x)[1]: 2.1f}  "
            out += "\n"

        print(out)

    p0 = TrialFunction(funspace_CG)
    p1 = TestFunction(funspace_CG)

    u0 = TrialFunction(funspace_Ned)

    A = npmat(inner(grad(p0), grad(p1))*dx)[np.ix_(ref_idx_CG, ref_idx_CG)]
    M = npmat(inner(u0, grad(p1))*dx)[np.ix_(ref_idx_CG, ref_idx_Ned)]


    B_edge = _map_edge(degree)
    B_face = np.zeros(shape=(ndof_CG, ndof_Ned))


    i1 = np.zeros(shape=(ndof_CG), dtype=bool)
    j1 = np.zeros(shape=(ndof_Ned), dtype=bool)
    
    for edge_i in range(3):
        i1[:] = 0; j1[:] = 0

        i1[3 + edge_i * ndof_CG_edge: 3 + (edge_i + 1) * ndof_CG_edge] = 1
        j1[edge_i * ndof_Ned_edge: (edge_i + 1) * ndof_Ned_edge] = 1

        # I have no clue where this factor "-2/sqrt(2)" comes from, but it's what makes it work
        B_face[np.ix_(i1, j1)] = -2/sqrt(2) * B_edge[2:, :] 


    i0 = np.zeros(shape=(ndof_CG), dtype=bool)
    i1[:] = 1; j1[:] = 1

    i0[ndof_CG - ndof_CG_face:] = 1
    i1[:3] = 0
    j1[0: 3*ndof_Ned_edge: ndof_Ned_edge] = 0

    B_face[np.ix_(i0, j1)] = np.linalg.inv(A[np.ix_(i0, i0)]) @ (M[np.ix_(i0, j1)] - A[i0, :] @ B_face[:, j1])

    #print(B_face)

    B = np.zeros_like(B_face)
    B[np.ix_(i1, j1)] = np.linalg.inv(A[np.ix_(i1, i1)]) @ M[np.ix_(i1, j1)]
    #print(B)

    return B_face


def _map_tetrahedron(degree):
    mesh = UnitTetrahedronMesh(0)

    funspace_RR = FunctionSpace(mesh, "R", 0)
    funspace_CG = FunctionSpace(mesh, "CG", degree)
    #funspace_Ned = FunctionSpace(mesh, "N1curl", degree)
    funspace_DG = FunctionSpace(mesh, "DG", degree-1)


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=125, suppress=True)
    _BOGOVSKII_DEBUG_ENABLE = True

    degree = 3
    B = map(degree, "face")