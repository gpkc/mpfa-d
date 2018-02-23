import numpy as np
from mesh_preprocessor import MeshManager
from pymoab import types

mesh = MeshManager('geometry_structured_test.msh')

K1 = np.array([1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0])

K2 = np.array([2.0, 0.0, 0.0,
               0.0, 2.0, 0.0,
               0.0, 0.0, 2.0])

mesh.set_media_property('Permeability', {1:K1, 2:K2}, dim_target=3)
mesh.set_boundary_condition('Dirichlet', {102:1.0, 101:0.0}, dim_target=2, set_nodes=True)
mesh.set_boundary_condition('Neumann', {201:0.0}, dim_target=2, set_nodes=True)

all_faces = mesh.mb.get_entities_by_dimension(0, 2)
print('ALL_FACES', len(all_faces))
all_tets = mesh.mb.get_entities_by_dimension(0, 3)
all_lines = mesh.mb.get_entities_by_dimension(0, 1)
all_vertex = mesh.mb.get_entities_by_dimension(0, 0)

entity_type = mesh.mb.type_from_handle(all_tets[0])
face_type = mesh.mb.type_from_handle(all_faces[0])
line_type = mesh.mb.type_from_handle(all_lines[0])
vertex_type = mesh.mb.type_from_handle(all_vertex[0])

print('Ent', entity_type)
print('face', face_type)
print('line', line_type)
print('vertex', vertex_type)

all_tets_press = mesh.mb.get_entities_by_type_and_tag(0, 9, mesh.perm_tag, np.array((None,)))
print("TETS_PERM", all_tets_press, len(all_tets_press))

all_nodes_neu = mesh.mb.get_entities_by_type_and_tag(0, types.MBVERTEX, mesh.neumann_tag, np.array((None,)))
print("NODES_NEU", all_nodes_neu, len(all_nodes_neu))

all_nodes_dirich = mesh.mb.get_entities_by_type_and_tag(0, types.MBVERTEX, mesh.dirichlet_tag, np.array((None,)))
print("NODES_DIRICH", all_nodes_dirich, len(all_nodes_dirich))
# for a_tet in all_tets_press:
#     nodes = mesh.mb.get_adjacencies(a_tet, 0)
#     print('TAMANHO', len(nodes))
