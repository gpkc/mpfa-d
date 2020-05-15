import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util


mb = core.Core()
mtu = topo_util.MeshTopoUtil(mb)

material_set_tag = mb.tag_get_handle(
    "MATERIAL_SET", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True
)

coords = np.array(
    [
        [0.000, 0.000, 0.000],
        [1.000, 0.000, 0.000],
        [0.000, 0.375, 0.000],
        [1.000, 0.575, 0.000],
        [0.000, 0.000, 1.000],
        [1.000, 0.000, 1.000],
        [0.000, 0.375, 1.000],
        [1.000, 0.575, 1.000],
        [1.000, 1.000, 0.000],
        [1.000, 0.580, 0.000],
        [0.000, 0.380, 0.000],
        [0.000, 1.000, 0.000],
        [0.000, 0.380, 1.000],
        [1.000, 0.580, 1.000],
        [0.000, 1.000, 1.000],
        [1.000, 1.000, 1.000],
    ]
)

verts = mb.create_vertices(coords.flatten())

# The Lower formation
tetra1 = mb.create_element(
    types.MBTET, [verts[0], verts[1], verts[3], verts[5]]
)
tetra2 = mb.create_element(
    types.MBTET, [verts[0], verts[4], verts[5], verts[6]]
)
tetra3 = mb.create_element(
    types.MBTET, [verts[0], verts[2], verts[3], verts[6]]
)
tetra4 = mb.create_element(
    types.MBTET, [verts[5], verts[6], verts[3], verts[7]]
)
tetra5 = mb.create_element(
    types.MBTET, [verts[0], verts[6], verts[3], verts[5]]
)

# The drain tetrahedra
tetra6 = mb.create_element(
    types.MBTET, [verts[6], verts[7], verts[3], verts[13]]
)
tetra7 = mb.create_element(
    types.MBTET, [verts[2], verts[3], verts[6], verts[10]]
)
tetra8 = mb.create_element(
    types.MBTET, [verts[6], verts[12], verts[10], verts[13]]
)
tetra9 = mb.create_element(
    types.MBTET, [verts[9], verts[3], verts[13], verts[10]]
)
tetra10 = mb.create_element(
    types.MBTET, [verts[13], verts[10], verts[3], verts[6]]
)

# The Upper formation
tetra11 = mb.create_element(
    types.MBTET, [verts[10], verts[8], verts[9], verts[13]]
)
tetra12 = mb.create_element(
    types.MBTET, [verts[10], verts[14], verts[13], verts[12]]
)
tetra13 = mb.create_element(
    types.MBTET, [verts[11], verts[14], verts[8], verts[10]]
)
tetra14 = mb.create_element(
    types.MBTET, [verts[15], verts[8], verts[14], verts[13]]
)
tetra15 = mb.create_element(
    types.MBTET, [verts[14], verts[10], verts[13], verts[8]]
)
all_verts = mb.get_entities_by_dimension(0, 0)
mtu.construct_aentities(all_verts)

formation_volumes = [
    tetra1,
    tetra2,
    tetra3,
    tetra4,
    tetra5,
    tetra11,
    tetra12,
    tetra13,
    tetra14,
    tetra15,
]
fracture_volumes = [tetra6, tetra7, tetra8, tetra9, tetra10]
formatin_volumes_set = mb.create_meshset()
for volume in formation_volumes:
    mb.add_entities(formatin_volumes_set, formation_volumes)
mb.tag_set_data(material_set_tag, formatin_volumes_set, 1)

fracture_volumes_set = mb.create_meshset()
for volume in formation_volumes:
    mb.add_entities(fracture_volumes_set, fracture_volumes)
mb.tag_set_data(material_set_tag, fracture_volumes_set, 2)

all_faces = mb.get_entities_by_dimension(0, 2)

faces = mb.get_entities_by_dimension(0, 2)
dirichlet_boundary_faces = mb.create_meshset()
neumann_boundary_faces = mb.create_meshset()
for face in faces:
    adjacent_vols = mtu.get_bridge_adjacencies(face, 2, 3)
    I, J, K = mtu.get_bridge_adjacencies(face, 2, 0)
    JI = mb.get_coords([I]) - mb.get_coords([J])
    JK = mb.get_coords([K]) - mb.get_coords([J])
    normal_area_vec = np.cross(JI, JK)
    # print(JI, JK, normal_area_vec)
    if len(adjacent_vols) < 2:
        if np.dot(normal_area_vec, np.array([0.0, 0.0, 1.0])) == 0.0:
            mb.add_entities(dirichlet_boundary_faces, [face])
        else:
            mb.add_entities(neumann_boundary_faces, [face])

mb.tag_set_data(material_set_tag, dirichlet_boundary_faces, 101)
mb.tag_set_data(material_set_tag, neumann_boundary_faces, 201)
volumes = mb.get_entities_by_dimension(0, 3)
ms = mb.create_meshset()
mb.add_entities(ms, volumes)
mb.write_file("louco.vtk",)
mb.write_file("meshes/mesh_slanted_mesh.h5m")
