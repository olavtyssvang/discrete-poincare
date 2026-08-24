#
#   For reference, this is the original code written specifically for the trimmed sequence
#       CG -> N1curl -> RT -> DG
# 
#   We hope it is easier to read than it abstraction.
#


import numpy as np
import scipy.sparse as sp
from firedrake import *


_BOGOVSKII_DEBUG_ENABLE = False


def _map_tetrahedron(degree):
    mesh = UnitTetrahedronMesh(0)

    funspace_RR = FunctionSpace(mesh, "R", 0)
    funspace_CG = FunctionSpace(mesh, "CG", degree)
    #funspace_N1 = FunctionSpace(mesh, "N1curl", degree)
    funspace_DG = FunctionSpace(mesh, "DG", degree-1)


def _debug_eval_edge_dof(degree):
    mesh = UnitIntervalMesh(1)

    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_CG = funspace_CG.finat_element.space_dimension()
    ndof_DG = funspace_DG.finat_element.space_dimension()

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_DG = funspace_DG.cell_node_map().values[0]

    out = "==== CG(edge):  ====\n"

    p = Function(funspace_CG)
    for i in range(ndof_CG):
        dof = np.zeros(shape=(ndof_CG))
        dof[i] = 1
        p.dat.data[ref_map_CG] = dof
        
        for x in [0, 1]:
            out += f"{p(x)[0]: 2.1f}  "
        out += "\n"

    out += "\n==== DG(edge):  ====\n"

    p = Function(funspace_DG)
    for i in range(ndof_DG):
        dof = np.zeros(shape=(ndof_DG))
        dof[i] = 1
        p.dat.data[ref_map_DG] = dof

        for x in [0, 1]:
            out += f"{p(x)[0]: 2.1f}  "
        out += "\n"

    print(out)

def _debug_eval_face_dof(degree):
    mesh = UnitTriangleMesh(0)

    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_N1 = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_CG, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_N1, ndof_N1_face, ndof_N1_edge = ndof_(funspace_N1)
    ndof_DG, ndof_DG_face = ndof_(funspace_DG)

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_N1 = funspace_N1.cell_node_map().values[0]
    ref_map_DG = funspace_DG.cell_node_map().values[0]

    out = "====  CG(face):  ====\n"

    p = Function(funspace_CG)
    for i in range(ndof_CG):
        dof = np.zeros(shape=(ndof_CG))
        dof[i] = 1
        p.dat.data[ref_map_CG] = dof
        
        for x in [(0, 0), (1, 0), (0, 1), (.5, 0), (0, .5), (.5, .5)]:
            out += f"{p(x): 2.1f}  "
        out += "\n"

    out += "\n==== Ned(face):  ====\n"

    p = Function(funspace_N1)
    for i in range(ndof_N1):
        dof = np.zeros(shape=(ndof_N1))
        dof[i] = 1
        p.dat.data[ref_map_N1] = dof

        for x in [(0, 0), (1, 0), (0, 1)]:
            out += f"{p(x)[0]: 2.1f},{p(x)[1]: 2.1f}  "
        out += "\n"

    out += "\n====  DG(face):  ====\n"

    p = Function(funspace_DG)
    for i in range(ndof_DG):
        dof = np.zeros(shape=(ndof_DG))
        dof[i] = 1
        p.dat.data[ref_map_DG] = dof

        for x in [(0, 0), (1, 0), (0, 1)]:
            out += f"{p(x): 2.1f}  "
        out += "\n"

    print(out)

def _debug_eval_tetrahedron_dof(degree):
    mesh = UnitTetrahedronMesh()

    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_N1 = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_RT = FunctionSpace(mesh, "RT", degree, variant="integral")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_CG, ndof_CG_tetrahedron, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_N1, ndof_N1_tetrahedron, ndof_N1_face, ndof_N1_edge = ndof_(funspace_N1)
    ndof_RT, ndof_RT_tetrahedron, ndof_RT_face = ndof_(funspace_RT)
    ndof_DG, ndof_DG_tetrahedron = ndof_(funspace_DG)

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_N1 = funspace_N1.cell_node_map().values[0]
    ref_map_RT = funspace_RT.cell_node_map().values[0]
    ref_map_DG = funspace_DG.cell_node_map().values[0]

    out = "====  CG(tetrahedron):  ====\n"

    p = Function(funspace_CG)
    for i in range(ndof_CG):
        dof = np.zeros(shape=(ndof_CG))
        dof[i] = 1
        p.dat.data[ref_map_CG] = dof
        
        for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 
                  (.5, 0, 0), (0, .5, 0), (0, 0, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5),
                  (1/3, 1/3, 0), (1/3, 0, 1/3), (0, 1/3, 1/3), (1/3, 1/3, 1/3)]:
            out += f"{p(x): 2.1f}  "
        out += "\n"

    out += "\n==== Ned(tetrahedron):  ====\n"

    p = Function(funspace_N1)
    for i in range(ndof_N1):
        dof = np.zeros(shape=(ndof_N1))
        dof[i] = 1
        p.dat.data[ref_map_N1] = dof

        for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 
                  (.5, 0, 0), (0, .5, 0), (0, 0, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]:
            out += f"{p(x)[0]: 2.1f},{p(x)[1]: 2.1f},{p(x)[2]: 2.1f}  "
        out += "\n"

    out += "\n====  RT(tetrahedron):  ====\n"

    p = Function(funspace_RT)
    for i in range(ndof_RT):
        dof = np.zeros(shape=(ndof_RT))
        dof[i] = 1
        p.dat.data[ref_map_RT] = dof

        for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            out += f"{p(x)[0]: 2.1f},{p(x)[1]: 2.1f},{p(x)[2]: 2.1f}  "
        out += "\n"

    out += "\n====  DG(tetrahedron):  ====\n"

    p = Function(funspace_DG)
    for i in range(ndof_DG):
        dof = np.zeros(shape=(ndof_DG))
        dof[i] = 1
        p.dat.data[ref_map_DG] = dof

        for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            out += f"{p(x): 2.1f}  "
        out += "\n"

    print(out)

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


_face_edges = enumerate([(0, 1, 2), (1, 3, 5), (0, 3, 4), (2, 4, 5)])

def _map_DG_edge(degree):
    mesh = UnitIntervalMesh(1)

    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_CG = funspace_CG.finat_element.space_dimension()
    ndof_DG = funspace_DG.finat_element.space_dimension()

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_DG = funspace_DG.cell_node_map().values[0]


    p0 = TrialFunction(funspace_CG)
    p1 = TestFunction(funspace_CG)
    q0 = TrialFunction(funspace_DG)

    LHS = npmat(inner(grad(p0), grad(p1))*dx)[np.ix_(ref_map_CG, ref_map_CG)]
    RHS = npmat(inner(q0, p1.dx(0))*dx)[np.ix_(ref_map_CG, ref_map_DG)]


    mask_CG_bubble = np.zeros(shape=(ndof_CG), dtype=bool)
    mask_CG_bubble[2:] = 1

    mask_DG = np.ones(shape=(ndof_DG), dtype=bool)
    mask_DG[0] = 0

    Bogovskii_DG_edge = np.zeros(shape=(ndof_CG, ndof_DG))
    Bogovskii_DG_edge[np.ix_(mask_CG_bubble, mask_DG)] \
        = np.linalg.inv(LHS[np.ix_(mask_CG_bubble, mask_CG_bubble)]) @ RHS[np.ix_(mask_CG_bubble, mask_DG)]

    return Bogovskii_DG_edge
    #
    #
def _map_N1_face(degree):
    mesh = UnitTriangleMesh(0)
    (u, v) = SpatialCoordinate(mesh)
    V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v, sqrt(3)/2 * v]))
    mesh = Mesh(coords)

    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_N1 = FunctionSpace(mesh, "N1curl", degree, variant="integral")

    ndof_CG, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_N1, ndof_N1_face, ndof_N1_edge = ndof_(funspace_N1)

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_N1 = funspace_N1.cell_node_map().values[0]


    p0 = TrialFunction(funspace_CG)
    p1 = TestFunction(funspace_CG)
    u0 = TrialFunction(funspace_N1)

    LHS = npmat(inner(grad(p0), grad(p1))*dx)[np.ix_(ref_map_CG, ref_map_CG)]
    RHS = npmat(inner(u0, grad(p1))*dx)[np.ix_(ref_map_CG, ref_map_N1)]


    mask_CG_bubble = np.zeros(shape=(ndof_CG), dtype=bool)
    mask_CG_bubble[ndof_CG - ndof_CG_face:] = 1

    mask_CG = np.zeros(shape=(ndof_CG), dtype=bool)
    mask_N1 = np.zeros(shape=(ndof_N1), dtype=bool)

    Bogovskii_DG_edge = _map_DG_edge(degree)
    Bogovskii_N1_face = np.zeros(shape=(ndof_CG, ndof_N1))
    

    for edge_i in range(3):
        mask_CG[:] = 0; mask_N1[:] = 0

        mask_CG[3 + edge_i * ndof_CG_edge: 3 + (edge_i + 1) * ndof_CG_edge] = 1
        mask_N1[edge_i * ndof_N1_edge: (edge_i + 1) * ndof_N1_edge] = 1

        # I have no clue where this factor "-2/sqrt(2)" comes from, but it's what makes it work
        Bogovskii_N1_face[np.ix_(mask_CG, mask_N1)] = -sqrt(2) * Bogovskii_DG_edge[2:, :] 


    mask_N1[:] = 1
    mask_N1[0: 3*ndof_N1_edge: ndof_N1_edge] = 0

    Bogovskii_N1_face[np.ix_(mask_CG_bubble, mask_N1)] \
        = np.linalg.inv(LHS[np.ix_(mask_CG_bubble, mask_CG_bubble)]) \
        @ (RHS[np.ix_(mask_CG_bubble, mask_N1)] - LHS[mask_CG_bubble, :] @ Bogovskii_N1_face[:, mask_N1])


    return Bogovskii_N1_face
    #
    #
def _map_DG_face(degree, variant):
    mesh = UnitTriangleMesh(0)

    if variant != "integral":
        (u, v) = SpatialCoordinate(mesh)
        V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=2)
        coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v, sqrt(3)/2 * v]))
        mesh = Mesh(coords)

    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_N1 = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant=variant)
    mixspace_CG_N1 = MixedFunctionSpace([funspace_CG, funspace_N1])

    ndof_CG, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_N1, ndof_N1_face, ndof_N1_edge = ndof_(funspace_N1)
    ndof_DG, ndof_DG_face = ndof_(funspace_DG)

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_N1 = funspace_N1.cell_node_map().values[0]
    ref_map_DG = funspace_DG.cell_node_map().values[0]
    
    ref_map_LHS = np.zeros(shape=(ndof_CG + ndof_N1), dtype=int)
    ref_map_LHS[:ndof_CG] = ref_map_CG
    ref_map_LHS[ndof_CG:] = ref_map_N1 + ndof_CG

    p0, u0 = TrialFunctions(mixspace_CG_N1)
    p1, u1 = TestFunctions(mixspace_CG_N1)
    q0 = TrialFunction(funspace_DG)

    LHS = npmat(
        inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx
    +   inner(p0, p1)*dx - inner(u0, grad(p1))*dx
    )[np.ix_(ref_map_LHS, ref_map_LHS)]

    RHS = npmat(inner(q0, curl(u1))*dx)[np.ix_(ref_map_LHS, ref_map_DG)]


    mask_CG_bubble = np.zeros(shape=(ndof_CG), dtype=bool)
    mask_CG_bubble[ndof_CG - ndof_CG_face:] = 1

    mask_N1_bubble = np.zeros(shape=(ndof_N1), dtype=bool)
    mask_N1_bubble[ndof_N1 - ndof_N1_face:] = 1

    mask_LHS = np.zeros(shape=(ndof_CG + ndof_N1), dtype=bool)
    mask_LHS[:ndof_CG] = mask_CG_bubble
    mask_LHS[ndof_CG:] = mask_N1_bubble

    mask_DG = np.ones(shape=(ndof_DG), dtype=bool)
    if variant == "integral":
        mask_DG[0] = 0

    Bogovskii_DG_face = np.zeros(shape=(ndof_N1, ndof_DG))
    Bogovskii_DG_face[np.ix_(mask_N1_bubble, mask_DG)] \
        = (np.linalg.inv(LHS[np.ix_(mask_LHS, mask_LHS)]) @ RHS[np.ix_(mask_LHS, mask_DG)])[np.sum(mask_CG_bubble):, :]

    return Bogovskii_DG_face
    #
    #
def _map_N1_tetrahedron(degree):
    mesh = UnitTetrahedronMesh()
    (u, v, w) = SpatialCoordinate(mesh)
    V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=3)
    coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v + 1/2 * w, sqrt(3)/2 * v + sqrt(3)/6 * w, sqrt(6)/3 * w]))
    mesh = Mesh(coords)
    
    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_N1 = FunctionSpace(mesh, "N1curl", degree, variant="integral")

    ndof_CG, ndof_CG_tetrahedron, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_N1, ndof_N1_tetrahedron, ndof_N1_face, ndof_N1_edge = ndof_(funspace_N1)

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_N1 = funspace_N1.cell_node_map().values[0]


    p0 = TrialFunction(funspace_CG)
    p1 = TestFunction(funspace_CG)
    u0 = TrialFunction(funspace_N1)

    LHS = npmat(inner(grad(p0), grad(p1))*dx)[np.ix_(ref_map_CG, ref_map_CG)]
    RHS = npmat(inner(u0, grad(p1))*dx)[np.ix_(ref_map_CG, ref_map_N1)]


    mask_CG_bubble = np.zeros(shape=(ndof_CG), dtype=bool)
    mask_CG_bubble[ndof_CG - ndof_CG_tetrahedron:] = 1

    mask_CG = np.zeros(shape=(ndof_CG), dtype=bool)
    mask_N1 = np.zeros(shape=(ndof_N1), dtype=bool)

    Bogovskii_N1_face = _map_N1_face(degree)
    Bogovskii_N1_tetrahedron = np.zeros(shape=(ndof_CG, ndof_N1))

    for face_i, edges in _face_edges:
        mask_CG[:] = 0; mask_N1[:] = 0

        for edge_i in edges:
            mask_CG[4 + edge_i * ndof_CG_edge: 4 + (edge_i + 1) * ndof_CG_edge] = 1
            mask_N1[edge_i * ndof_N1_edge: (edge_i + 1) * ndof_N1_edge] = 1

        mask_CG[4 + 6*ndof_CG_edge + face_i * ndof_CG_face: 4 + 6*ndof_CG_edge + (face_i + 1) * ndof_CG_face] = 1
        mask_N1[6*ndof_N1_edge + face_i * ndof_N1_face: 6*ndof_N1_edge + (face_i + 1) * ndof_N1_face] = 1

        Bogovskii_N1_tetrahedron[np.ix_(mask_CG, mask_N1)] = -sqrt(8) * Bogovskii_N1_face[3:, :]


    mask_N1[:] = 1
    mask_N1[0: 3*ndof_N1_edge: ndof_N1_edge] = 0

    Bogovskii_N1_tetrahedron[np.ix_(mask_CG_bubble, mask_N1)] \
        = np.linalg.inv(LHS[np.ix_(mask_CG_bubble, mask_CG_bubble)]) \
        @ (RHS[np.ix_(mask_CG_bubble, mask_N1)] - LHS[mask_CG_bubble, :] @ Bogovskii_N1_tetrahedron[:, mask_N1])

    return Bogovskii_N1_tetrahedron
    #
    #
def _map_RT_tetrahedron(degree):
    mesh = UnitTetrahedronMesh()
    (u, v, w) = SpatialCoordinate(mesh)
    V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=3)
    coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v + 1/2 * w, sqrt(3)/2 * v + sqrt(3)/6 * w, sqrt(6)/3 * w]))
    mesh = Mesh(coords)

    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_N1 = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_RT = FunctionSpace(mesh, "RT", degree, variant="integral")
    mixspace_CG_N1 = MixedFunctionSpace([funspace_CG, funspace_N1])

    ndof_CG, ndof_CG_tetrahedron, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_N1, ndof_N1_tetrahedron, ndof_N1_face, ndof_N1_edge = ndof_(funspace_N1)
    ndof_RT, ndof_RT_tetrahedron, ndof_RT_face = ndof_(funspace_RT)

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_N1 = funspace_N1.cell_node_map().values[0]
    ref_map_RT = funspace_RT.cell_node_map().values[0]
    
    ref_map_LHS = np.zeros(shape=(ndof_CG + ndof_N1), dtype=int)
    ref_map_LHS[:ndof_CG] = ref_map_CG
    ref_map_LHS[ndof_CG:] = ref_map_N1 + ndof_CG

    p0, u0 = TrialFunctions(mixspace_CG_N1)
    p1, u1 = TestFunctions(mixspace_CG_N1)
    v0 = TrialFunction(funspace_RT)

    LHS = npmat(
        inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx
    +   inner(p0, p1)*dx - inner(u0, grad(p1))*dx
    )[np.ix_(ref_map_LHS, ref_map_LHS)]

    RHS = npmat(inner(v0, curl(u1))*dx)[np.ix_(ref_map_LHS, ref_map_RT)]


    mask_CG_bubble = np.zeros(shape=(ndof_CG), dtype=bool)
    mask_CG_bubble[ndof_CG - ndof_CG_tetrahedron:] = 1

    mask_N1_bubble = np.zeros(shape=(ndof_N1), dtype=bool)
    mask_N1_bubble[ndof_N1 - ndof_N1_tetrahedron:] = 1

    mask_LHS = np.zeros(shape=(ndof_CG + ndof_N1), dtype=bool)
    mask_LHS[:ndof_CG] = mask_CG_bubble
    mask_LHS[ndof_CG:] = mask_N1_bubble

    mask_N1 = np.zeros(shape=(ndof_N1), dtype=bool)
    mask_RT = np.zeros(shape=(ndof_RT), dtype=bool)

    Bogovskii_DG_face = _map_DG_face(degree, variant="integral")
    Bogovskii_RT_tetrahedron = np.zeros(shape=(ndof_N1, ndof_RT))

    for face_i in range(4):
        mask_N1[:] = 0; mask_RT[:] = 0

        mask_N1[6*ndof_N1_edge + face_i * ndof_N1_face: 6*ndof_N1_edge + (face_i + 1) * ndof_N1_face] = 1
        mask_RT[face_i * ndof_RT_face: (face_i + 1) * ndof_RT_face] = 1

        Bogovskii_RT_tetrahedron[np.ix_(mask_N1, mask_RT)] = Bogovskii_DG_face[3*ndof_N1_edge:, :]

    mask_RT[:] = 1
    mask_RT[0: 4*ndof_RT_face: ndof_RT_face] = 0

    Bogovskii_RT_tetrahedron[np.ix_(mask_N1_bubble, mask_RT)] \
        = (np.linalg.inv(LHS[np.ix_(mask_LHS, mask_LHS)]) \
        @ (RHS[np.ix_(mask_LHS, mask_RT)] - LHS[mask_LHS, ndof_CG:] @ Bogovskii_RT_tetrahedron[:, mask_RT]))[np.sum(mask_CG_bubble):, :]

    return Bogovskii_RT_tetrahedron
    #
    #
def _map_DG_tetrahedron(degree, variant):
    mesh = UnitTetrahedronMesh()
    (u, v, w) = SpatialCoordinate(mesh)
    V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=3)
    coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v + 1/2 * w, sqrt(3)/2 * v + sqrt(3)/6 * w, sqrt(6)/3 * w]))
    mesh = Mesh(coords)

    funspace_N1 = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_RT = FunctionSpace(mesh, "RT", degree, variant="integral")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant=variant)
    mixspace_N1_RT = MixedFunctionSpace([funspace_N1, funspace_RT])

    ndof_N1, ndof_N1_tetrahedron, ndof_N1_face, ndof_N1_edge = ndof_(funspace_N1)
    ndof_RT, ndof_RT_tetrahedron, ndof_RT_face = ndof_(funspace_RT)
    ndof_DG, ndof_DG_tetrahedron = ndof_(funspace_DG)


    ref_map_N1 = funspace_N1.cell_node_map().values[0]
    ref_map_RT = funspace_RT.cell_node_map().values[0]
    ref_map_DG = funspace_DG.cell_node_map().values[0]
    
    ref_map_LHS = np.zeros(shape=(ndof_N1 + ndof_RT), dtype=int)
    ref_map_LHS[:ndof_N1] = ref_map_N1
    ref_map_LHS[ndof_N1:] = ref_map_RT + ndof_N1

    u0, v0 = TrialFunctions(mixspace_N1_RT)
    u1, v1 = TestFunctions(mixspace_N1_RT)
    q0 = TrialFunction(funspace_DG)

    LHS = npmat(
        inner(curl(u0), v1)*dx + inner(div(v0), div(v1))*dx
    +   inner(u0, u1)*dx - inner(v0, curl(u1))*dx
    )[np.ix_(ref_map_LHS, ref_map_LHS)]

    RHS = npmat(inner(q0, div(v1))*dx)[np.ix_(ref_map_LHS, ref_map_DG)]


    mask_N1_bubble = np.zeros(shape=(ndof_N1), dtype=bool)
    mask_N1_bubble[ndof_N1 - ndof_N1_tetrahedron:] = 1

    mask_RT_bubble = np.zeros(shape=(ndof_RT), dtype=bool)
    mask_RT_bubble[ndof_RT - ndof_RT_tetrahedron:] = 1

    mask_LHS = np.zeros(shape=(ndof_N1 + ndof_RT), dtype=bool)
    mask_LHS[:ndof_N1] = mask_N1_bubble
    mask_LHS[ndof_N1:] = mask_RT_bubble

    mask_DG = np.ones(shape=(ndof_DG), dtype=bool)
    if variant == "integral":
        mask_DG[0] = 0

    Bogovskii_DG_tetrahedron = np.zeros(shape=(ndof_RT, ndof_DG))
    Bogovskii_DG_tetrahedron[np.ix_(mask_RT_bubble, mask_DG)] \
        = (np.linalg.inv(LHS[np.ix_(mask_LHS, mask_LHS)]) @ RHS[np.ix_(mask_LHS, mask_DG)])[np.sum(mask_N1_bubble):, :]

    return sqrt(2) * Bogovskii_DG_tetrahedron
    #
    #





if __name__ == "__main__":
    #np.set_printoptions(precision=2, linewidth=125, suppress=True)
    _BOGOVSKII_DEBUG_ENABLE = True

    #_debug_eval_tetrahedron_dof(2)

    degree = 3
    B = _map_RT_tetrahedron(degree)

    mesh = UnitTetrahedronMesh()
    (x, y, z) = SpatialCoordinate(mesh)

    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_N1 = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_RT = FunctionSpace(mesh, "RT", degree, variant="integral")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="spectral")

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_N1 = funspace_N1.cell_node_map().values[0]
    ref_map_RT = funspace_RT.cell_node_map().values[0]
    ref_map_DG = funspace_DG.cell_node_map().values[0]

    ndof_CG, ndof_CG_tetrahedron, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_N1, ndof_N1_tetrahedron, ndof_N1_face, ndof_N1_edge = ndof_(funspace_N1)
    ndof_RT, ndof_RT_tetrahedron, ndof_RT_face = ndof_(funspace_RT)
    ndof_DG, ndof_DG_tetrahedron = ndof_(funspace_DG)


    funspace_RT1 = FunctionSpace(mesh, "RT", 1, variant="integral(6)")

    u0 = Function(funspace_N1).interpolate(as_vector([sin(pi * (x)), sin(pi/2 * (x)), sin(pi/4 * (x))]))
    If = Function(funspace_RT1).interpolate(curl(u0))
    f = Function(funspace_RT).interpolate(curl(u0) - If)

    u = Function(funspace_N1)
    u.dat.data[ref_map_N1] = B @ f.dat.data[ref_map_RT]
    du = Function(funspace_RT).interpolate(curl(u))

    print(f.dat.data[ref_map_RT][:ndof_RT - ndof_RT_tetrahedron])
    print(du.dat.data[ref_map_RT][:ndof_RT - ndof_RT_tetrahedron])
    print()
    print(f.dat.data[ref_map_RT][ndof_RT - ndof_RT_tetrahedron:])
    print(du.dat.data[ref_map_RT][ndof_RT - ndof_RT_tetrahedron:])
    print()
    print(np.max(np.abs(f.dat.data[:] - du.dat.data[:])))

if False:

    f = Function(funspace_N1).interpolate(as_vector([sin(pi * y) * sin(pi * (1-x-y)), sin(pi/2 * x) * sin(pi/2 * (1-x-y))]))
    q = Function(funspace_DG).interpolate(sin(x-y))


    u = Function(funspace_N1)
    u.dat.data[ref_map_N1] = B @ q.dat.data[ref_map_DG]


    ndof_N1 = funspace_N1.finat_element.space_dimension()
    ndof_DG = funspace_DG.finat_element.space_dimension()

    domain = f"{{[i,j]: 0 <= i < {ndof_N1} and 0 <= j < {ndof_DG}}}"
    instructions = f"""
    for i
        a[i] = 0.0
        for j
            a[i] = a[i] + T[i*{ndof_DG} + j] * b[j]
        end
    end
    """

    par_loop(
        (domain, instructions),
        dx,
        {"T": (Constant(B), READ), "b": (q, READ), "a": (u, WRITE)}
    )


    curl_u = Function(funspace_DG).interpolate(curl(u))
    print(curl_u.dat.data[:])
    print(q.dat.data[:]) 