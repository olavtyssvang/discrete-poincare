import numpy as np
import scipy.sparse as sp
from firedrake import *


_BOGOVSKII_DEBUG_ENABLE = False


def _constructor_args(funspace):
    el = funspace.ufl_element()

    space = el.sobolev_space
    family = el.family()
    degree = el.degree()
    variant = el.variant()

    # targeting custom quadrature degrees, when variant="integral(...)"
    if len(variant) > 8 and variant[:8] == "integral":
        variant = "integral" 

    return space, family, degree, variant

def bogovskii_map(funspace_from, funspace_to):

    mesh = funspace_from.mesh()
    assert mesh is funspace_to.mesh(), "Mesh is ambiguous."

    dim = mesh.topological_dimension()
    from_space, from_family, from_degree, from_variant = _constructor_args(funspace_from)
    to_space, to_family, to_degree, to_variant = _constructor_args(funspace_to)


    # Check that the pairing of spaces makes sense 

    if from_family == "Nedelec 1st kind H(curl)":
        assert to_space == ufl.sobolevspace.H1, "Invalid pairing of spaces. Expected H(curl) -> H1."
        assert to_family == "Lagrange", "Unsupported pairing of families."
        assert to_degree == from_degree, "Invalid pairing of degrees."

        assert from_variant == "integral", "Unsupported H(curl) variant. Expected integral."
        assert to_variant == "equispaced" or to_variant == "spectral", "Unsupported H1 variant. Expected equispaced or spectral."

        if dim == 2:
            return _map_HCurl_face([to_family, from_family], [to_degree, from_degree], [to_variant, from_variant])
            #
        if dim == 3:
            return _map_HCurl_tetrahedron([to_family, from_family], [to_degree, from_degree], [to_variant, from_variant])
        #
    if from_family == "Nedelec 2nd kind H(curl)":
        assert to_space == ufl.sobolevspace.H1, "Invalid pairing of spaces. Expected H(curl) -> H1."
        assert to_family == "Lagrange", "Unsupported pairing of families."
        assert to_degree == from_degree + 1, "Invalid pairing of degrees."

        assert from_variant == "integral", "Unsupported H(curl) variant. Expected integral."
        assert to_variant == "equispaced" or to_variant == "spectral", "Unsupported H1 variant. Expected equispaced or spectral."

        if dim == 2:
            return _map_HCurl_face([to_family, from_family], [to_degree, from_degree], [to_variant, from_variant])
            #
        if dim == 3:
            return _map_HCurl_tetrahedron([to_family, from_family], [to_degree, from_degree], [to_variant, from_variant])
        #
    if from_family == "Raviart-Thomas":
        if dim == 2:
            assert to_space == ufl.sobolevspace.H1, f"Invalid pairing of spaces. Expected H(div) -> H1."
            assert to_family == "Lagrange", "Unsupported pairing of families."
            assert to_degree == from_degree, "Invalid pairing of degrees."

            assert from_variant == "integral", "Unsupported H(div) variant. Expected integral."
            assert to_variant == "equispaced" or to_variant == "spectral", "Unsupported H1 variant. Expected equispaced or spectral." 

            return _map_HDiv_face([to_family, from_family], [to_degree, from_degree], [to_variant, from_variant])
            #
        if dim == 3:
            assert to_space == ufl.sobolevspace.HCurl, "Invalid pairing of spaces. Expected H(div) -> H(curl)."
            assert to_family == "Nedelec 1st kind H(curl)" or to_family == "Nedelec 2nd kind H(curl)", "Unsupported pairing of families."
            assert to_degree == from_degree, "Invalid pairing of degrees."

            assert from_variant == "integral", "Unsupported H(div) variant. Expected integral."
            assert to_variant == "integral", "Unsupported H(curl) variant. Expected integral."

            if to_family == "Nedelec 1st kind H(curl)":
                return _map_HDiv_tetrahedron(["Lagrange", to_family, from_family], [to_degree, to_degree, from_degree], ["equispaced", to_variant, from_variant])
                #
            elif to_family == "Nedelec 2nd kind H(curl)":
                return _map_HDiv_tetrahedron(["Lagrange", to_family, from_family], [to_degree + 1, to_degree, from_degree], ["equispaced", to_variant, from_variant])
        #
    if from_family == "Brezzi-Douglas-Marini":
        if dim == 2:
            assert to_space == ufl.sobolevspace.H1, f"Invalid pairing of spaces. Expected H(div) -> H1."
            assert to_family == "Lagrange", "Unsupported pairing of families."
            assert to_degree == from_degree + 1, "Invalid pairing of degrees."

            assert from_variant == "integral", "Unsupported H(div) variant. Expected integral."
            assert to_variant == "equispaced" or to_variant == "spectral", "Unsupported H1 variant. Expected equispaced or spectral." 

            return _map_HDiv_face([to_family, from_family], [to_degree, from_degree], [to_variant, from_variant])
            #
        if dim == 3:
            assert to_space == ufl.sobolevspace.HCurl, "Invalid pairing of spaces. Expected H(div) -> H(curl)."
            assert to_family == "Nedelec 1st kind H(curl)" or to_family == "Nedelec 2nd kind H(curl)", "Unsupported pairing of families."
            assert to_degree == from_degree + 1, "Invalid pairing of degrees."

            assert from_variant == "integral", "Unsupported H(div) variant. Expected integral."
            assert to_variant == "integral", "Unsupported H(curl) variant. Expected integral."

            if to_family == "Nedelec 1st kind H(curl)":
                return _map_HDiv_tetrahedron(["Lagrange", to_family, from_family], [to_degree, to_degree, from_degree], ["equispaced", to_variant, from_variant])
            elif to_family == "Nedelec 2nd kind H(curl)":
                return _map_HDiv_tetrahedron(["Lagrange", to_family, from_family], [to_degree + 1, to_degree, from_degree], ["equispaced", to_variant, from_variant])
        #
    if from_family == "Discontinuous Lagrange":
        if dim == 1:
            assert to_space == ufl.sobolevspace.H1, "Invalid pairing of spaces. Expected L2 -> H1."
            assert to_family == "Lagrange", "Unsupported pairing of families."
            assert to_degree == from_degree + 1, "Invalid pairing of degrees." 

            assert from_variant == "equispaced" or from_variant == "spectral", "Unsupported L2 variant. Expected equispaced or spectral."
            assert to_variant == "integral" or to_variant == "spectral", "Unsupported H1 variant. Expected equispaced or spectral."

            return _map_L2_edge([to_family, from_family], [to_degree, from_degree], [to_variant, from_variant])
            #
        if dim == 2:
            assert to_space == ufl.sobolevspace.HCurl or to_space == ufl.sobolevspace.HDiv, "Invalid pairing of spaces. Expected L2 -> H(curl) or L2 -> H(div)."
            assert from_variant == "equispaced" or from_variant == "spectral", "Unsupported L2 variant. Expected equispaced or spectral."

            if to_space == ufl.sobolevspace.HCurl:
                assert to_family == "Nedelec 1st kind H(curl)" or to_family == "Nedelec 2nd kind H(curl)", "Unsupported pairing of families."
                assert to_degree == from_degree + 1, "Invalid pairing of degrees." 
                assert to_variant == "integral", "Unsupported H(curl) variant. Expected integral."
                #
            elif to_space == ufl.sobolevspace.HDiv:
                assert to_family == "Raviart-Thomas" or to_family == "Brezzi-Douglas-Marini", "Unsupported pairing of families."
                assert to_degree == from_degree + 1, "Invalid pairing of degrees." 
                assert to_variant == "integral", "Unsupported H(div) variant. Expected integral."

            if to_family == "Nedelec 1st kind H(curl)":
                return _map_L2_HCurl_face(["Lagrange", to_family, from_family], [to_degree, to_degree, from_degree], ["integral", to_variant, from_variant])
            elif to_family == "Nedelec 2nd kind H(curl)":
                return _map_L2_HCurl_face(["Lagrange", to_family, from_family], [to_degree + 1, to_degree, from_degree], ["integral", to_variant, from_variant])
            if to_family == "Raviart-Thomas":
                return _map_L2_HDiv_face(["Lagrange", to_family, from_family], [to_degree, to_degree, from_degree], ["integral", to_variant, from_variant])
            elif to_family == "Brezzi-Douglas-Marini":
                return _map_L2_HDiv_face(["Lagrange", to_family, from_family], [to_degree + 1, to_degree, from_degree], ["integral", to_variant, from_variant])
            #  
        if dim == 3:
            assert to_space == ufl.sobolevspace.HDiv, "Invalid pairing of spaces. Expected L2 -> H(div)."
            assert to_family == "Raviart-Thomas" or to_family == "Brezzi-Douglas-Marini", "Unsupported pairing of families."
            assert to_degree == from_degree + 1, "Invalid pairing of degrees." 

            assert from_variant == "equispaced" or from_variant == "spectral", "Unsupported L2 variant. Expected equispaced or spectral."
            assert to_variant == "integral", "Unsupported H(div) variant. Expected integral."

            if to_family == "Raviart-Thomas":
                return _map_L2_tetrahedron(["Nedelec 1st kind H(curl)", to_family, from_family], [to_degree, to_degree, from_degree], ["integral", to_variant, from_variant])
            elif to_family == "Brezzi-Douglas-Marini":
                return _map_L2_tetrahedron(["Nedelec 1st kind H(curl)", to_family, from_family], [to_degree + 1, to_degree, from_degree], ["integral", to_variant, from_variant])
        #
    else:
        raise NotImplementedError
    

def _debug_eval_edge_dof(degree):
    mesh = UnitIntervalMesh(1)

    funspace_H1 = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_L2 = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_H1 = funspace_H1.finat_element.space_dimension()
    ndof_L2 = funspace_L2.finat_element.space_dimension()

    ref_map_H1 = funspace_H1.cell_node_map().values[0]
    ref_map_L2 = funspace_L2.cell_node_map().values[0]

    out = "==== CG(edge):  ====\n"

    p = Function(funspace_H1)
    for i in range(ndof_H1):
        dof = np.zeros(shape=(ndof_H1))
        dof[i] = 1
        p.dat.data[ref_map_H1] = dof
        
        for x in [0, 1]:
            out += f"{p(x)[0]: 2.1f}  "
        out += "\n"

    out += "\n==== DG(edge):  ====\n"

    p = Function(funspace_L2)
    for i in range(ndof_L2):
        dof = np.zeros(shape=(ndof_L2))
        dof[i] = 1
        p.dat.data[ref_map_L2] = dof

        for x in [0, 1]:
            out += f"{p(x)[0]: 2.1f}  "
        out += "\n"

    print(out)
    #
def _debug_eval_face_dof(degree):
    mesh = UnitTriangleMesh(0)

    funspace_H1 = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_HCurl = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_L2 = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_H1, ndof_H1_face, ndof_H1_edge, ndof_H1_vert = ndof_(funspace_H1)
    ndof_HCurl, ndof_HCurl_face, ndof_HCurl_edge = ndof_(funspace_HCurl)
    ndof_L2, ndof_L2_face = ndof_(funspace_L2)

    ref_map_H1 = funspace_H1.cell_node_map().values[0]
    ref_map_HCurl = funspace_HCurl.cell_node_map().values[0]
    ref_map_L2 = funspace_L2.cell_node_map().values[0]

    out = "====  CG(face):  ====\n"

    p = Function(funspace_H1)
    for i in range(ndof_H1):
        dof = np.zeros(shape=(ndof_H1))
        dof[i] = 1
        p.dat.data[ref_map_H1] = dof
        
        for x in [(0, 0), (1, 0), (0, 1), (.5, 0), (0, .5), (.5, .5)]:
            out += f"{p(x): 2.1f}  "
        out += "\n"

    out += "\n==== Ned(face):  ====\n"

    p = Function(funspace_HCurl)
    for i in range(ndof_HCurl):
        dof = np.zeros(shape=(ndof_HCurl))
        dof[i] = 1
        p.dat.data[ref_map_HCurl] = dof

        for x in [(0, 0), (1, 0), (0, 1)]:
            out += f"{p(x)[0]: 2.1f},{p(x)[1]: 2.1f}  "
        out += "\n"

    out += "\n====  DG(face):  ====\n"

    p = Function(funspace_L2)
    for i in range(ndof_L2):
        dof = np.zeros(shape=(ndof_L2))
        dof[i] = 1
        p.dat.data[ref_map_L2] = dof

        for x in [(0, 0), (1, 0), (0, 1)]:
            out += f"{p(x): 2.1f}  "
        out += "\n"

    print(out)
    #
def _debug_eval_tetrahedron_dof(degree):
    mesh = UnitTetrahedronMesh()

    funspace_H1 = FunctionSpace(mesh, "CG", degree, variant="equispaced")
    funspace_HCurl = FunctionSpace(mesh, "N1curl", degree, variant="integral")
    funspace_HDiv = FunctionSpace(mesh, "RT", degree, variant="integral")
    funspace_L2 = FunctionSpace(mesh, "DG", degree-1, variant="integral")

    ndof_H1, ndof_H1_tetrahedron, ndof_H1_face, ndof_H1_edge, ndof_H1_vert = ndof_(funspace_H1)
    ndof_HCurl, ndof_HCurl_tetrahedron, ndof_HCurl_face, ndof_HCurl_edge = ndof_(funspace_HCurl)
    ndof_HDiv, ndof_HDiv_tetrahedron, ndof_HDiv_face = ndof_(funspace_HDiv)
    ndof_L2, ndof_L2_tetrahedron = ndof_(funspace_L2)

    ref_map_H1 = funspace_H1.cell_node_map().values[0]
    ref_map_HCurl = funspace_HCurl.cell_node_map().values[0]
    ref_map_HDiv = funspace_HDiv.cell_node_map().values[0]
    ref_map_L2 = funspace_L2.cell_node_map().values[0]

    out = "====  CG(tetrahedron):  ====\n"

    p = Function(funspace_H1)
    for i in range(ndof_H1):
        dof = np.zeros(shape=(ndof_H1))
        dof[i] = 1
        p.dat.data[ref_map_H1] = dof
        
        for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 
                  (.5, 0, 0), (0, .5, 0), (0, 0, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5),
                  (1/3, 1/3, 0), (1/3, 0, 1/3), (0, 1/3, 1/3), (1/3, 1/3, 1/3)]:
            out += f"{p(x): 2.1f}  "
        out += "\n"

    out += "\n==== Ned(tetrahedron):  ====\n"

    p = Function(funspace_HCurl)
    for i in range(ndof_HCurl):
        dof = np.zeros(shape=(ndof_HCurl))
        dof[i] = 1
        p.dat.data[ref_map_HCurl] = dof

        for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 
                  (.5, 0, 0), (0, .5, 0), (0, 0, .5), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]:
            out += f"{p(x)[0]: 2.1f},{p(x)[1]: 2.1f},{p(x)[2]: 2.1f}  "
        out += "\n"

    out += "\n====  RT(tetrahedron):  ====\n"

    p = Function(funspace_HDiv)
    for i in range(ndof_HDiv):
        dof = np.zeros(shape=(ndof_HDiv))
        dof[i] = 1
        p.dat.data[ref_map_HDiv] = dof

        for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            out += f"{p(x)[0]: 2.1f},{p(x)[1]: 2.1f},{p(x)[2]: 2.1f}  "
        out += "\n"

    out += "\n====  DG(tetrahedron):  ====\n"

    p = Function(funspace_L2)
    for i in range(ndof_L2):
        dof = np.zeros(shape=(ndof_L2))
        dof[i] = 1
        p.dat.data[ref_map_L2] = dof

        for x in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            out += f"{p(x): 2.1f}  "
        out += "\n"

    print(out)
    #


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


def _map_L2_edge(family, degree, variant):
    mesh = UnitIntervalMesh(1)

    funspace_H1 = FunctionSpace(mesh, family[0], degree[0], variant=variant[0])
    funspace_L2 = FunctionSpace(mesh, family[1], degree[1], variant=variant[1])

    ndof_H1 = funspace_H1.finat_element.space_dimension()
    ndof_L2 = funspace_L2.finat_element.space_dimension()

    ref_map_H1 = funspace_H1.cell_node_map().values[0]
    ref_map_L2 = funspace_L2.cell_node_map().values[0]


    p0 = TrialFunction(funspace_H1)
    p1 = TestFunction(funspace_H1)
    q0 = TrialFunction(funspace_L2)

    LHS = npmat(inner(grad(p0), grad(p1))*dx)[np.ix_(ref_map_H1, ref_map_H1)]
    RHS = npmat(inner(q0, p1.dx(0))*dx)[np.ix_(ref_map_H1, ref_map_L2)]


    mask_H1_bubble = np.zeros(shape=(ndof_H1), dtype=bool)
    mask_H1_bubble[2:] = 1

    mask_L2 = np.ones(shape=(ndof_L2), dtype=bool)
    mask_L2[0] = 0

    Bogovskii_L2_edge = np.zeros(shape=(ndof_H1, ndof_L2))
    Bogovskii_L2_edge[np.ix_(mask_H1_bubble, mask_L2)] \
        = np.linalg.inv(LHS[np.ix_(mask_H1_bubble, mask_H1_bubble)]) @ RHS[np.ix_(mask_H1_bubble, mask_L2)]

    return Bogovskii_L2_edge
    #
    #
def _map_HCurl_face(family, degree, variant):
    mesh = UnitTriangleMesh(0)
    (u, v) = SpatialCoordinate(mesh)
    V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v, sqrt(3)/2 * v]))
    mesh = Mesh(coords)

    funspace_H1 = FunctionSpace(mesh, family[0], degree[0], variant=variant[0])
    funspace_HCurl = FunctionSpace(mesh, family[1], degree[1], variant=variant[1])

    ndof_H1, ndof_H1_face, ndof_H1_edge, ndof_H1_vert = ndof_(funspace_H1)
    ndof_HCurl, ndof_HCurl_face, ndof_HCurl_edge = ndof_(funspace_HCurl)

    ref_map_H1 = funspace_H1.cell_node_map().values[0]
    ref_map_HCurl = funspace_HCurl.cell_node_map().values[0]


    p0 = TrialFunction(funspace_H1)
    p1 = TestFunction(funspace_H1)
    u0 = TrialFunction(funspace_HCurl)

    LHS = npmat(inner(grad(p0), grad(p1))*dx)[np.ix_(ref_map_H1, ref_map_H1)]
    RHS = npmat(inner(u0, grad(p1))*dx)[np.ix_(ref_map_H1, ref_map_HCurl)]


    mask_H1_bubble = np.zeros(shape=(ndof_H1), dtype=bool)
    mask_H1_bubble[ndof_H1 - ndof_H1_face:] = 1

    mask_H1 = np.zeros(shape=(ndof_H1), dtype=bool)
    mask_HCurl = np.zeros(shape=(ndof_HCurl), dtype=bool)

    Bogovskii_L2_edge = None
    if family[1] == "Nedelec 1st kind H(curl)":
        Bogovskii_L2_edge = _map_L2_edge([family[0], "Discontinuous Lagrange"], [degree[0], degree[1] - 1], [variant[0], "integral"])
    elif family[1] == "Nedelec 2nd kind H(curl)":
        Bogovskii_L2_edge = _map_L2_edge([family[0], "Discontinuous Lagrange"], [degree[0], degree[1]], [variant[0], "integral"])
    else:
        raise NotImplementedError

    Bogovskii_HCurl_face = np.zeros(shape=(ndof_H1, ndof_HCurl))
    

    for edge_i in range(3):
        mask_H1[:] = 0; mask_HCurl[:] = 0

        mask_H1[3 + edge_i * ndof_H1_edge: 3 + (edge_i + 1) * ndof_H1_edge] = 1
        mask_HCurl[edge_i * ndof_HCurl_edge: (edge_i + 1) * ndof_HCurl_edge] = 1

        # I have no clue where this factor "-2/sqrt(2)" comes from, but it's what makes it work
        Bogovskii_HCurl_face[np.ix_(mask_H1, mask_HCurl)] = -sqrt(2) * Bogovskii_L2_edge[2:, :] 


    mask_HCurl[:] = 1
    mask_HCurl[0: 3*ndof_HCurl_edge: ndof_HCurl_edge] = 0

    Bogovskii_HCurl_face[np.ix_(mask_H1_bubble, mask_HCurl)] \
        = np.linalg.inv(LHS[np.ix_(mask_H1_bubble, mask_H1_bubble)]) \
        @ (RHS[np.ix_(mask_H1_bubble, mask_HCurl)] - LHS[mask_H1_bubble, :] @ Bogovskii_HCurl_face[:, mask_HCurl])


    return Bogovskii_HCurl_face
    #
    #
def _map_HDiv_face(family, degree, variant):
    mesh = UnitTriangleMesh(0)
    (u, v) = SpatialCoordinate(mesh)
    V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v, sqrt(3)/2 * v]))
    mesh = Mesh(coords)

    funspace_H1 = FunctionSpace(mesh, family[0], degree[0], variant=variant[0])
    funspace_HDiv = FunctionSpace(mesh, family[1], degree[1], variant=variant[1])

    ndof_H1, ndof_H1_face, ndof_H1_edge, ndof_H1_vert = ndof_(funspace_H1)
    ndof_HDiv, ndof_HDiv_face, ndof_HDiv_edge = ndof_(funspace_HDiv)

    ref_map_H1 = funspace_H1.cell_node_map().values[0]
    ref_map_HDiv = funspace_HDiv.cell_node_map().values[0]


    p0 = TrialFunction(funspace_H1)
    p1 = TestFunction(funspace_H1)
    u0 = TrialFunction(funspace_HDiv)

    LHS = npmat(inner(curl(p0), curl(p1))*dx)[np.ix_(ref_map_H1, ref_map_H1)]
    RHS = npmat(inner(u0, curl(p1))*dx)[np.ix_(ref_map_H1, ref_map_HDiv)]


    mask_H1_bubble = np.zeros(shape=(ndof_H1), dtype=bool)
    mask_H1_bubble[ndof_H1 - ndof_H1_face:] = 1

    mask_H1 = np.zeros(shape=(ndof_H1), dtype=bool)
    mask_HDiv = np.zeros(shape=(ndof_HDiv), dtype=bool)

    Bogovskii_L2_edge = None
    if family[1] == "Raviart-Thomas":
        Bogovskii_L2_edge = _map_L2_edge(["Lagrange", "Discontinuous Lagrange"], [degree[1], degree[1] - 1], ["equispaced", "integral"])
    elif family[1] == "Brezzi-Douglas-Marini":
        Bogovskii_L2_edge = _map_L2_edge(["Lagrange", "Discontinuous Lagrange"], [degree[1] + 1, degree[1]], ["equispaced", "integral"])
    else:
        raise NotImplementedError

    Bogovskii_HDiv_face = np.zeros(shape=(ndof_H1, ndof_HDiv))
    

    for edge_i in range(3):
        mask_H1[:] = 0; mask_HDiv[:] = 0

        mask_H1[3 + edge_i * ndof_H1_edge: 3 + (edge_i + 1) * ndof_H1_edge] = 1
        mask_HDiv[edge_i * ndof_HDiv_edge: (edge_i + 1) * ndof_HDiv_edge] = 1

        # I have no clue where this factor "-2/sqrt(2)" comes from, but it's what makes it work
        Bogovskii_HDiv_face[np.ix_(mask_H1, mask_HDiv)] = -sqrt(2) * Bogovskii_L2_edge[2:, :] 


    mask_HDiv[:] = 1
    mask_HDiv[0: 3*ndof_HDiv_edge: ndof_HDiv_edge] = 0

    Bogovskii_HDiv_face[np.ix_(mask_H1_bubble, mask_HDiv)] \
        = np.linalg.inv(LHS[np.ix_(mask_H1_bubble, mask_H1_bubble)]) \
        @ (RHS[np.ix_(mask_H1_bubble, mask_HDiv)] - LHS[mask_H1_bubble, :] @ Bogovskii_HDiv_face[:, mask_HDiv])


    return Bogovskii_HDiv_face
    #
    #
def _map_L2_HCurl_face(family, degree, variant):
    mesh = UnitTriangleMesh(0)

    if variant[2] != "integral":
        (u, v) = SpatialCoordinate(mesh)
        V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=2)
        coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v, sqrt(3)/2 * v]))
        mesh = Mesh(coords)

    funspace_H1 = FunctionSpace(mesh, family[0], degree[0], variant=variant[0])
    funspace_HCurl = FunctionSpace(mesh, family[1], degree[1], variant=variant[1])
    funspace_L2 = FunctionSpace(mesh, family[2], degree[2], variant=variant[2])
    mixspace_H1_HCurl = MixedFunctionSpace([funspace_H1, funspace_HCurl])

    ndof_H1, ndof_H1_face, ndof_H1_edge, ndof_H1_vert = ndof_(funspace_H1)
    ndof_HCurl, ndof_HCurl_face, ndof_HCurl_edge = ndof_(funspace_HCurl)
    ndof_L2, ndof_L2_face = ndof_(funspace_L2)

    ref_map_H1 = funspace_H1.cell_node_map().values[0]
    ref_map_HCurl = funspace_HCurl.cell_node_map().values[0]
    ref_map_L2 = funspace_L2.cell_node_map().values[0]
    
    ref_map_LHS = np.zeros(shape=(ndof_H1 + ndof_HCurl), dtype=int)
    ref_map_LHS[:ndof_H1] = ref_map_H1
    ref_map_LHS[ndof_H1:] = ref_map_HCurl + ndof_H1

    p0, u0 = TrialFunctions(mixspace_H1_HCurl)
    p1, u1 = TestFunctions(mixspace_H1_HCurl)
    q0 = TrialFunction(funspace_L2)

    LHS = npmat(
        inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx
    +   inner(p0, p1)*dx - inner(u0, grad(p1))*dx
    )[np.ix_(ref_map_LHS, ref_map_LHS)]

    RHS = npmat(inner(q0, curl(u1))*dx)[np.ix_(ref_map_LHS, ref_map_L2)]


    mask_H1_bubble = np.zeros(shape=(ndof_H1), dtype=bool)
    mask_H1_bubble[ndof_H1 - ndof_H1_face:] = 1

    mask_HCurl_bubble = np.zeros(shape=(ndof_HCurl), dtype=bool)
    mask_HCurl_bubble[ndof_HCurl - ndof_HCurl_face:] = 1

    mask_LHS = np.zeros(shape=(ndof_H1 + ndof_HCurl), dtype=bool)
    mask_LHS[:ndof_H1] = mask_H1_bubble
    mask_LHS[ndof_H1:] = mask_HCurl_bubble

    mask_L2 = np.ones(shape=(ndof_L2), dtype=bool)
    if variant[2] == "integral":
        mask_L2[0] = 0

    Bogovskii_L2_face = np.zeros(shape=(ndof_HCurl, ndof_L2))
    Bogovskii_L2_face[np.ix_(mask_HCurl_bubble, mask_L2)] \
        = (np.linalg.inv(LHS[np.ix_(mask_LHS, mask_LHS)]) @ RHS[np.ix_(mask_LHS, mask_L2)])[np.sum(mask_H1_bubble):, :]

    if variant[2] != "integral":
        return 2/sqrt(3)* Bogovskii_L2_face
    else:
        return Bogovskii_L2_face
    #
    #
def _map_L2_HDiv_face(family, degree, variant):
    mesh = UnitTriangleMesh(0)

    if variant[2] != "integral":
        (u, v) = SpatialCoordinate(mesh)
        V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=2)
        coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v, sqrt(3)/2 * v]))
        mesh = Mesh(coords)

    funspace_H1 = FunctionSpace(mesh, family[0], degree[0], variant=variant[0])
    funspace_HDiv = FunctionSpace(mesh, family[1], degree[1], variant=variant[1])
    funspace_L2 = FunctionSpace(mesh, family[2], degree[2], variant=variant[2])
    mixspace_H1_HDiv = MixedFunctionSpace([funspace_H1, funspace_HDiv])

    ndof_H1, ndof_H1_face, ndof_H1_edge, ndof_H1_vert = ndof_(funspace_H1)
    ndof_HDiv, ndof_HDiv_face, ndof_HDiv_edge = ndof_(funspace_HDiv)
    ndof_L2, ndof_L2_face = ndof_(funspace_L2)

    ref_map_H1 = funspace_H1.cell_node_map().values[0]
    ref_map_HDiv = funspace_HDiv.cell_node_map().values[0]
    ref_map_L2 = funspace_L2.cell_node_map().values[0]
    
    ref_map_LHS = np.zeros(shape=(ndof_H1 + ndof_HDiv), dtype=int)
    ref_map_LHS[:ndof_H1] = ref_map_H1
    ref_map_LHS[ndof_H1:] = ref_map_HDiv + ndof_H1

    p0, u0 = TrialFunctions(mixspace_H1_HDiv)
    p1, u1 = TestFunctions(mixspace_H1_HDiv)
    q0 = TrialFunction(funspace_L2)

    LHS = npmat(
        inner(curl(p0), u1)*dx + inner(div(u0), div(u1))*dx
    +   inner(p0, p1)*dx - inner(u0, curl(p1))*dx
    )[np.ix_(ref_map_LHS, ref_map_LHS)]

    RHS = npmat(inner(q0, div(u1))*dx)[np.ix_(ref_map_LHS, ref_map_L2)]


    mask_H1_bubble = np.zeros(shape=(ndof_H1), dtype=bool)
    mask_H1_bubble[ndof_H1 - ndof_H1_face:] = 1

    mask_HDiv_bubble = np.zeros(shape=(ndof_HDiv), dtype=bool)
    mask_HDiv_bubble[ndof_HDiv - ndof_HDiv_face:] = 1

    mask_LHS = np.zeros(shape=(ndof_H1 + ndof_HDiv), dtype=bool)
    mask_LHS[:ndof_H1] = mask_H1_bubble
    mask_LHS[ndof_H1:] = mask_HDiv_bubble

    mask_L2 = np.ones(shape=(ndof_L2), dtype=bool)
    if variant[2] == "integral":
        mask_L2[0] = 0

    Bogovskii_L2_face = np.zeros(shape=(ndof_HDiv, ndof_L2))
    Bogovskii_L2_face[np.ix_(mask_HDiv_bubble, mask_L2)] \
        = (np.linalg.inv(LHS[np.ix_(mask_LHS, mask_LHS)]) @ RHS[np.ix_(mask_LHS, mask_L2)])[np.sum(mask_H1_bubble):, :]

    if variant[2] != "integral":
        return 2/sqrt(3) * Bogovskii_L2_face
    else:
        return Bogovskii_L2_face
    #
    #
def _map_HCurl_tetrahedron(family, degree, variant):
    mesh = UnitTetrahedronMesh()
    (u, v, w) = SpatialCoordinate(mesh)
    V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=3)
    coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v + 1/2 * w, sqrt(3)/2 * v + sqrt(3)/6 * w, sqrt(6)/3 * w]))
    mesh = Mesh(coords)
    
    funspace_H1 = FunctionSpace(mesh, family[0], degree[0], variant=variant[0])
    funspace_HCurl = FunctionSpace(mesh, family[1], degree[1], variant=variant[1])

    ndof_H1, ndof_H1_tetrahedron, ndof_H1_face, ndof_H1_edge, ndof_H1_vert = ndof_(funspace_H1)
    ndof_HCurl, ndof_HCurl_tetrahedron, ndof_HCurl_face, ndof_HCurl_edge = ndof_(funspace_HCurl)

    ref_map_H1 = funspace_H1.cell_node_map().values[0]
    ref_map_HCurl = funspace_HCurl.cell_node_map().values[0]


    p0 = TrialFunction(funspace_H1)
    p1 = TestFunction(funspace_H1)
    u0 = TrialFunction(funspace_HCurl)

    LHS = npmat(inner(grad(p0), grad(p1))*dx)[np.ix_(ref_map_H1, ref_map_H1)]
    RHS = npmat(inner(u0, grad(p1))*dx)[np.ix_(ref_map_H1, ref_map_HCurl)]


    mask_H1_bubble = np.zeros(shape=(ndof_H1), dtype=bool)
    mask_H1_bubble[ndof_H1 - ndof_H1_tetrahedron:] = 1

    mask_H1 = np.zeros(shape=(ndof_H1), dtype=bool)
    mask_HCurl = np.zeros(shape=(ndof_HCurl), dtype=bool)

    Bogovskii_HCurl_face = _map_HCurl_face(family, degree, variant)
    Bogovskii_HCurl_tetrahedron = np.zeros(shape=(ndof_H1, ndof_HCurl))

    for face_i, edges in enumerate([(0, 1, 2), (0, 3, 4), (1, 3, 5), (2, 4, 5)]):
        mask_H1[:] = 0; mask_HCurl[:] = 0

        for edge_i in edges:
            mask_H1[4 + edge_i * ndof_H1_edge: 4 + (edge_i + 1) * ndof_H1_edge] = 1
            mask_HCurl[edge_i * ndof_HCurl_edge: (edge_i + 1) * ndof_HCurl_edge] = 1

        mask_H1[4 + 6*ndof_H1_edge + face_i * ndof_H1_face: 4 + 6*ndof_H1_edge + (face_i + 1) * ndof_H1_face] = 1
        mask_HCurl[6*ndof_HCurl_edge + face_i * ndof_HCurl_face: 6*ndof_HCurl_edge + (face_i + 1) * ndof_HCurl_face] = 1

        Bogovskii_HCurl_tetrahedron[np.ix_(mask_H1, mask_HCurl)] = Bogovskii_HCurl_face[3:, :]


    mask_HCurl[:] = 1
    mask_HCurl[0: 4*ndof_HCurl_edge: ndof_HCurl_edge] = 0

    Bogovskii_HCurl_tetrahedron[np.ix_(mask_H1_bubble, mask_HCurl)] \
        = np.linalg.inv(LHS[np.ix_(mask_H1_bubble, mask_H1_bubble)]) \
        @ (RHS[np.ix_(mask_H1_bubble, mask_HCurl)] - LHS[mask_H1_bubble, :] @ Bogovskii_HCurl_tetrahedron[:, mask_HCurl])

    return Bogovskii_HCurl_tetrahedron
    #
    #
def _map_HDiv_tetrahedron(family, degree, variant):
    mesh = UnitTetrahedronMesh()
    (u, v, w) = SpatialCoordinate(mesh)
    V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=3)
    coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v + 1/2 * w, sqrt(3)/2 * v + sqrt(3)/6 * w, sqrt(6)/3 * w]))
    mesh = Mesh(coords)

    funspace_H1 = FunctionSpace(mesh, family[0], degree[0], variant=variant[0])
    funspace_HCurl = FunctionSpace(mesh, family[1], degree[1], variant=variant[1])
    funspace_HDiv = FunctionSpace(mesh, family[2], degree[2], variant=variant[2])
    mixspace_H1_HCurl = MixedFunctionSpace([funspace_H1, funspace_HCurl])

    ndof_H1, ndof_H1_tetrahedron, ndof_H1_face, ndof_H1_edge, ndof_H1_vert = ndof_(funspace_H1)
    ndof_HCurl, ndof_HCurl_tetrahedron, ndof_HCurl_face, ndof_HCurl_edge = ndof_(funspace_HCurl)
    ndof_HDiv, ndof_HDiv_tetrahedron, ndof_HDiv_face = ndof_(funspace_HDiv)

    ref_map_H1 = funspace_H1.cell_node_map().values[0]
    ref_map_HCurl = funspace_HCurl.cell_node_map().values[0]
    ref_map_HDiv = funspace_HDiv.cell_node_map().values[0]
    
    ref_map_LHS = np.zeros(shape=(ndof_H1 + ndof_HCurl), dtype=int)
    ref_map_LHS[:ndof_H1] = ref_map_H1
    ref_map_LHS[ndof_H1:] = ref_map_HCurl + ndof_H1

    p0, u0 = TrialFunctions(mixspace_H1_HCurl)
    p1, u1 = TestFunctions(mixspace_H1_HCurl)
    v0 = TrialFunction(funspace_HDiv)

    LHS = npmat(
        inner(grad(p0), u1)*dx + inner(curl(u0), curl(u1))*dx
    +   inner(p0, p1)*dx - inner(u0, grad(p1))*dx
    )[np.ix_(ref_map_LHS, ref_map_LHS)]

    RHS = npmat(inner(v0, curl(u1))*dx)[np.ix_(ref_map_LHS, ref_map_HDiv)]


    mask_H1_bubble = np.zeros(shape=(ndof_H1), dtype=bool)
    mask_H1_bubble[ndof_H1 - ndof_H1_tetrahedron:] = 1

    mask_HCurl_bubble = np.zeros(shape=(ndof_HCurl), dtype=bool)
    mask_HCurl_bubble[ndof_HCurl - ndof_HCurl_tetrahedron:] = 1

    mask_LHS = np.zeros(shape=(ndof_H1 + ndof_HCurl), dtype=bool)
    mask_LHS[:ndof_H1] = mask_H1_bubble
    mask_LHS[ndof_H1:] = mask_HCurl_bubble

    mask_HCurl = np.zeros(shape=(ndof_HCurl), dtype=bool)
    mask_HDiv = np.zeros(shape=(ndof_HDiv), dtype=bool)

    Bogovskii_L2_face = None
    if family[2] == "Raviart-Thomas":
        Bogovskii_L2_face = _map_L2_HCurl_face([family[0], family[1], "Discontinuous Lagrange"], [degree[0], degree[1], degree[2] - 1], [variant[0], variant[1], "integral"])
    elif family[2] == "Brezzi-Douglas-Marini":
        Bogovskii_L2_face = _map_L2_HCurl_face([family[0], family[1], "Discontinuous Lagrange"], [degree[0], degree[1], degree[2]], [variant[0], variant[1], "integral"])
    else:
        raise NotImplementedError

    Bogovskii_HDiv_tetrahedron = np.zeros(shape=(ndof_HCurl, ndof_HDiv))

    for face_i in range(4):
        mask_HCurl[:] = 0; mask_HDiv[:] = 0

        mask_HCurl[6*ndof_HCurl_edge + face_i * ndof_HCurl_face: 6*ndof_HCurl_edge + (face_i + 1) * ndof_HCurl_face] = 1
        mask_HDiv[face_i * ndof_HDiv_face: (face_i + 1) * ndof_HDiv_face] = 1

        Bogovskii_HDiv_tetrahedron[np.ix_(mask_HCurl, mask_HDiv)] = -sqrt(8) * Bogovskii_L2_face[3*ndof_HCurl_edge:, :]

    mask_HDiv[:] = 1
    mask_HDiv[0: 4*ndof_HDiv_face: ndof_HDiv_face] = 0

    Bogovskii_HDiv_tetrahedron[np.ix_(mask_HCurl_bubble, mask_HDiv)] \
        = (np.linalg.inv(LHS[np.ix_(mask_LHS, mask_LHS)]) \
        @ (RHS[np.ix_(mask_LHS, mask_HDiv)] - LHS[mask_LHS, ndof_H1:] @ Bogovskii_HDiv_tetrahedron[:, mask_HDiv]))[np.sum(mask_H1_bubble):, :]

    return Bogovskii_HDiv_tetrahedron
    #
    #
def _map_L2_tetrahedron(family, degree, variant):
    mesh = UnitTetrahedronMesh()
    (u, v, w) = SpatialCoordinate(mesh)
    V_coord = VectorFunctionSpace(mesh, "CG", 1, dim=3)
    coords = Function(V_coord).interpolate(as_vector([u + 1/2 * v + 1/2 * w, sqrt(3)/2 * v + sqrt(3)/6 * w, sqrt(6)/3 * w]))
    mesh = Mesh(coords)

    funspace_HCurl = FunctionSpace(mesh, family[0], degree[0], variant=variant[0])
    funspace_HDiv = FunctionSpace(mesh, family[1], degree[1], variant=variant[1])
    funspace_L2 = FunctionSpace(mesh, family[2], degree[2], variant=variant[2])
    mixspace_HCurl_HDiv = MixedFunctionSpace([funspace_HCurl, funspace_HDiv])

    ndof_HCurl, ndof_HCurl_tetrahedron, ndof_HCurl_face, ndof_HCurl_edge = ndof_(funspace_HCurl)
    ndof_HDiv, ndof_HDiv_tetrahedron, ndof_HDiv_face = ndof_(funspace_HDiv)
    ndof_L2, ndof_L2_tetrahedron = ndof_(funspace_L2)


    ref_map_HCurl = funspace_HCurl.cell_node_map().values[0]
    ref_map_HDiv = funspace_HDiv.cell_node_map().values[0]
    ref_map_L2 = funspace_L2.cell_node_map().values[0]
    
    ref_map_LHS = np.zeros(shape=(ndof_HCurl + ndof_HDiv), dtype=int)
    ref_map_LHS[:ndof_HCurl] = ref_map_HCurl
    ref_map_LHS[ndof_HCurl:] = ref_map_HDiv + ndof_HCurl

    u0, v0 = TrialFunctions(mixspace_HCurl_HDiv)
    u1, v1 = TestFunctions(mixspace_HCurl_HDiv)
    q0 = TrialFunction(funspace_L2)

    LHS = npmat(
        inner(curl(u0), v1)*dx + inner(div(v0), div(v1))*dx
    +   inner(u0, u1)*dx - inner(v0, curl(u1))*dx
    )[np.ix_(ref_map_LHS, ref_map_LHS)]

    RHS = npmat(inner(q0, div(v1))*dx)[np.ix_(ref_map_LHS, ref_map_L2)]


    mask_HCurl_bubble = np.zeros(shape=(ndof_HCurl), dtype=bool)
    mask_HCurl_bubble[ndof_HCurl - ndof_HCurl_tetrahedron:] = 1

    mask_HDiv_bubble = np.zeros(shape=(ndof_HDiv), dtype=bool)
    mask_HDiv_bubble[ndof_HDiv - ndof_HDiv_tetrahedron:] = 1

    mask_LHS = np.zeros(shape=(ndof_HCurl + ndof_HDiv), dtype=bool)
    mask_LHS[:ndof_HCurl] = mask_HCurl_bubble
    mask_LHS[ndof_HCurl:] = mask_HDiv_bubble

    mask_L2 = np.ones(shape=(ndof_L2), dtype=bool)
    if variant == "integral":
        mask_L2[0] = 0

    Bogovskii_L2_tetrahedron = np.zeros(shape=(ndof_HDiv, ndof_L2))
    Bogovskii_L2_tetrahedron[np.ix_(mask_HDiv_bubble, mask_L2)] \
        = (np.linalg.inv(LHS[np.ix_(mask_LHS, mask_LHS)]) @ RHS[np.ix_(mask_LHS, mask_L2)])[np.sum(mask_HCurl_bubble):, :]

    return sqrt(2) * Bogovskii_L2_tetrahedron
    #
    #
    