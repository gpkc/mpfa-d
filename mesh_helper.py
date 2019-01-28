import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util


mb = core.Core()
mtu = topo_util.MeshTopoUtil(mb)

dirichlet_boundaries_tag = mb.tag_get_handle(
    "Dirichlet faces", 1, types.MB_TYPE_INTEGER,
    types.MB_TAG_SPARSE, True)

material_set_tag = mb.tag_get_handle(
    "MATERIAL_SET", 1, types.MB_TYPE_INTEGER,
    types.MB_TAG_SPARSE, True)

coords = np.array([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [1.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [1.0, 0.0, 1.0],
                   [0.0, 1.0, 1.0],
                   [1.0, 1.0, 1.0]])

verts = mb.create_vertices(coords.flatten())

tetra1 = mb.create_element(types.MBTET, [verts[0], verts[1],
                                         verts[3], verts[5]])
tetra2 = mb.create_element(types.MBTET, [verts[0], verts[4],
                                         verts[5], verts[6]])
tetra3 = mb.create_element(types.MBTET, [verts[0], verts[2],
                                         verts[3], verts[6]])
tetra4 = mb.create_element(types.MBTET, [verts[5], verts[6],
                                         verts[3], verts[7]])
tetra5 = mb.create_element(types.MBTET, [verts[0], verts[6],
                                         verts[3], verts[5]])
all_volumes = [tetra1, tetra2, tetra3, tetra4, tetra5]
mb.tag_set_data(material_set_tag, all_volumes, np.repeat(1, len(all_volumes)))

all_verts = mb.get_entities_by_dimension(0, 0)
mtu.construct_aentities(all_verts)

all_faces = mb.get_entities_by_dimension(0, 2)

dirichlet_boundary_faces = []
faces = mb.get_entities_by_dimension(0, 2)
dirichlet_boundary_faces = mb.create_meshset()
for face in faces:
    adjacent_vols = mtu.get_bridge_adjacencies(face, 2, 3)
    if len(adjacent_vols) < 2:
        mb.add_entities(dirichlet_boundary_faces, [face])
mb.tag_set_data(material_set_tag, dirichlet_boundary_faces, 101)

mb.write_file('test_mesh_5_vols.h5m')
