import numpy as np
import scipy.sparse as sp
from firedrake import *


_BOGOVSKII_DEBUG_ENABLE = False

def map(degree, domain):
    if domain == "edge":
        return _map_DG_edge(degree)
    
    if domain == "face":
        return _map_Ned_face(degree)
    
    if domain == "tetrahedron":
        return _map_tetrahedron(degree)
    
    raise ValueError(domain)

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
    funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_CG, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_Ned, ndof_Ned_face, ndof_Ned_edge = ndof_(funspace_Ned)
    ndof_DG, ndof_DG_face = ndof_(funspace_DG)

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_Ned = funspace_Ned.cell_node_map().values[0]
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

    p = Function(funspace_Ned)
    for i in range(ndof_Ned):
        dof = np.zeros(shape=(ndof_Ned))
        dof[i] = 1
        p.dat.data[ref_map_Ned] = dof

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
    funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_RT = FunctionSpace(mesh, "Raviart-Thomas", degree, variant="integral")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_CG, ndof_CG_tetrahedron, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_Ned, ndof_Ned_tetrahedron, ndof_Ned_face, ndof_Ned_edge = ndof_(funspace_Ned)
    ndof_RT, ndof_RT_tetrahedron, ndof_RT_face = ndof_(funspace_RT)
    ndof_DG, ndof_DG_tetrahedron = ndof_(funspace_DG)

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_Ned = funspace_Ned.cell_node_map().values[0]
    ref_map_RT = funspace_RT.cell_node_map().values[0]
    ref_map_DG = funspace_DG.cell_node_map().values[0]

    out = "====  CG(face):  ====\n"

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

    out += "\n==== Ned(face):  ====\n"

    p = Function(funspace_Ned)
    for i in range(ndof_Ned):
        dof = np.zeros(shape=(ndof_Ned))
        dof[i] = 1
        p.dat.data[ref_map_Ned] = dof

        for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 
                  (.5, 0, 0), (0, .5, 0), (0, 0, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]:
            out += f"{p(x)[0]: 2.1f},{p(x)[1]: 2.1f},{p(x)[2]: 2.1f}  "
        out += "\n"

    out += "\n====  RT(face):  ====\n"

    p = Function(funspace_RT)
    for i in range(ndof_RT):
        dof = np.zeros(shape=(ndof_RT))
        dof[i] = 1
        p.dat.data[ref_map_RT] = dof

        for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            out += f"{p(x)[0]: 2.1f},{p(x)[1]: 2.1f},{p(x)[2]: 2.1f}  "
        out += "\n"

    out += "\n====  DG(face):  ====\n"

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


def _map_Ned_face(degree):
    mesh = UnitTriangleMesh(0)

    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant="integral")

    ndof_CG, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_Ned, ndof_Ned_face, ndof_Ned_edge = ndof_(funspace_Ned)

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_Ned = funspace_Ned.cell_node_map().values[0]


    p0 = TrialFunction(funspace_CG)
    p1 = TestFunction(funspace_CG)
    u0 = TrialFunction(funspace_Ned)

    LHS = npmat(inner(grad(p0), grad(p1))*dx)[np.ix_(ref_map_CG, ref_map_CG)]
    RHS = npmat(inner(u0, grad(p1))*dx)[np.ix_(ref_map_CG, ref_map_Ned)]


    mask_CG_bubble = np.zeros(shape=(ndof_CG), dtype=bool)
    mask_CG_bubble[ndof_CG - ndof_CG_face:] = 1

    mask_CG = np.zeros(shape=(ndof_CG), dtype=bool)
    mask_Ned = np.zeros(shape=(ndof_Ned), dtype=bool)

    Bogovskii_DG_edge = _map_DG_edge(degree)
    Bogovskii_Ned_face = np.zeros(shape=(ndof_CG, ndof_Ned))
    

    for edge_i in range(3):
        mask_CG[:] = 0; mask_Ned[:] = 0

        mask_CG[3 + edge_i * ndof_CG_edge: 3 + (edge_i + 1) * ndof_CG_edge] = 1
        mask_Ned[edge_i * ndof_Ned_edge: (edge_i + 1) * ndof_Ned_edge] = 1

        # I have no clue where this factor "-2/sqrt(2)" comes from, but it's what makes it work
        Bogovskii_Ned_face[np.ix_(mask_CG, mask_Ned)] = -2/sqrt(2) * Bogovskii_DG_edge[2:, :] 


    mask_Ned[:] = 1
    mask_Ned[0: 3*ndof_Ned_edge: ndof_Ned_edge] = 0

    Bogovskii_Ned_face[np.ix_(mask_CG_bubble, mask_Ned)] \
        = np.linalg.inv(LHS[np.ix_(mask_CG_bubble, mask_CG_bubble)]) \
        @ (RHS[np.ix_(mask_CG_bubble, mask_Ned)] - LHS[mask_CG_bubble, :] @ Bogovskii_Ned_face[:, mask_Ned])


    return Bogovskii_Ned_face


def _map_DG_face(degree):
    mesh = UnitTriangleMesh(0)

    funspace_CG = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="integral")
    mixspace_CG_Ned = MixedFunctionSpace([funspace_CG, funspace_Ned])

    ndof_CG, ndof_CG_face, ndof_CG_edge, ndof_CG_vert = ndof_(funspace_CG)
    ndof_Ned, ndof_Ned_face, ndof_Ned_edge = ndof_(funspace_Ned)
    ndof_DG, ndof_DG_face = ndof_(funspace_DG)

    ref_map_CG = funspace_CG.cell_node_map().values[0]
    ref_map_Ned = funspace_Ned.cell_node_map().values[0]
    ref_map_DG = funspace_DG.cell_node_map().values[0]
    
    ref_map_LHS = np.zeros(shape=(ndof_CG + ndof_Ned), dtype=int)
    ref_map_LHS[:ndof_CG] = ref_map_CG
    ref_map_LHS[ndof_CG:] = ref_map_Ned + ndof_CG

    p0, u0 = TrialFunctions(mixspace_CG_Ned)
    p1, u1 = TestFunctions(mixspace_CG_Ned)
    q0 = TrialFunction(funspace_DG)

    LHS = npmat(
        inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx
    +   inner(p0, p1)*dx - inner(u0, grad(p1))*dx
    )[np.ix_(ref_map_LHS, ref_map_LHS)]

    RHS = npmat(inner(q0, curl(u1))*dx)[np.ix_(ref_map_LHS, ref_map_DG)]

    u0 = TrialFunction(funspace_Ned)
    u1 = TestFunction(funspace_Ned)

    print(LHS)
    print(npmat(inner(curl(u0), curl(u1))*dx)[np.ix_(ref_map_Ned, ref_map_Ned)])

    #print(RHS)

    mask_CG_bubble = np.zeros(shape=(ndof_CG), dtype=bool)
    mask_CG_bubble[ndof_CG - ndof_CG_face:] = 1

    mask_Ned_bubble = np.zeros(shape=(ndof_Ned), dtype=bool)
    mask_Ned_bubble[ndof_Ned - ndof_Ned_face:] = 1

    mask_LHS = np.zeros(shape=(ndof_CG + ndof_Ned), dtype=bool)
    mask_LHS[:ndof_CG] = mask_CG_bubble
    mask_LHS[ndof_CG:] = mask_Ned_bubble

    mask_DG = np.ones(shape=(ndof_DG), dtype=bool)
    mask_DG[0] = 0

    Bogovskii_DG_face = np.zeros(shape=(ndof_Ned, ndof_DG))
    Bogovskii_DG_face[np.ix_(mask_Ned_bubble, mask_DG)] \
        = (np.linalg.inv(LHS[np.ix_(mask_LHS, mask_LHS)]) @ RHS[np.ix_(mask_LHS, mask_DG)])[np.sum(mask_CG_bubble):, :]

    #Bogovskii_DG_face = (np.linalg.inv(LHS) @ RHS)[ndof_CG:, :]
    #Bogovskii_DG_face[:, 0] = 0

    return Bogovskii_DG_face

def _map_tetrahedron(degree):
    mesh = UnitTetrahedronMesh(0)

    funspace_RR = FunctionSpace(mesh, "R", 0)
    funspace_CG = FunctionSpace(mesh, "CG", degree)
    #funspace_Ned = FunctionSpace(mesh, "N1curl", degree)
    funspace_DG = FunctionSpace(mesh, "DG", degree-1)


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=125, suppress=True)
    _BOGOVSKII_DEBUG_ENABLE = True

    degree = 2
    B = _map_DG_face(degree)


    mesh = UnitTriangleMesh(0)
    (x, y) = SpatialCoordinate(mesh)

    funspace_Ned = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_DG = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ref_map_Ned = funspace_Ned.cell_node_map().values[0]
    ref_map_DG = funspace_DG.cell_node_map().values[0]

    f = Function(funspace_Ned).interpolate(as_vector([sin(pi * y) * sin(pi * (1-x-y)), sin(pi/2 * x) * sin(pi/2 * (1-x-y))]))
    q = Function(funspace_DG).interpolate(sin(x-y))

    print(norm(q, "L2"))

    u = Function(funspace_Ned)
    u.dat.data[ref_map_Ned] = B @ q.dat.data[ref_map_DG]


    ndof_Ned = funspace_Ned.finat_element.space_dimension()
    ndof_DG = funspace_DG.finat_element.space_dimension()

    domain = f"{{[i,j]: 0 <= i < {ndof_Ned} and 0 <= j < {ndof_DG}}}"
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