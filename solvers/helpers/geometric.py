# from pymoab import core
# from pymoab import topo_util
import numpy as np

# mb = core.Core()
# root_set = mb.get_root_set()
# mtu = topo_util.MeshTopoUtil(mb)


def point_distance(coords_1, coords_2):
    dist_vector = coords_1 - coords_2
    distance = np.sqrt(np.dot(dist_vector, dist_vector))
    return distance


# lru cache
def cached_area(face, node):

    # pega nos da face
    # pega coords do node
    # chama area vector
    pass


def _area_vector(nodes, ref_node=np.zeros(3), norma=False):
    ref_vect = nodes[0] - ref_node
    AB = nodes[1] - nodes[0]
    AC = nodes[2] - nodes[0]
    area_vector = np.cross(AB, AC) / 2.0
    if norma:
        area = np.sqrt(np.dot(area_vector, area_vector))
        return area
    if np.dot(area_vector, ref_vect) < 0.0:
        area_vector = -area_vector
        return [area_vector, -1]
    return [area_vector, 1]


def get_height(normal_vector, distance_vector):
    return np.absolute(
        np.dot(normal_vector, distance_vector)
        / np.sqrt(np.dot(normal_vector, normal_vector))
    )


def get_tetra_volume(tet_nodes):
    vect_1 = tet_nodes[1] - tet_nodes[0]
    vect_2 = tet_nodes[2] - tet_nodes[0]
    vect_3 = tet_nodes[3] - tet_nodes[0]
    vol_eval = abs(np.dot(np.cross(vect_1, vect_2), vect_3)) / 6.0
    return vol_eval
