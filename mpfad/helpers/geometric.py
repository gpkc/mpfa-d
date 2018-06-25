from math import sqrt
from pymoab import core
from pymoab import topo_util

import numpy as np

mb = core.Core()
root_set = mb.get_root_set()
mtu = topo_util.MeshTopoUtil(mb)


def point_distance(coords_1, coords_2):
    dist_vector = coords_1 - coords_2
    distance = sqrt(np.dot(dist_vector, dist_vector))
    return distance


def is_coplanar(vec_1, vec_2, vec_3):
    """ returns true or false if a set of 4 points are coplanar"""
    coplanar = np.dot(vec_1, np.cross(vec_2, vec_3))
    if coplanar == 0:
        return True
    else:
        return False


def get_tetra_volume(verts):

    pass


def _area_vector(nodes, ref_node):
    ref_vect = nodes[0] - ref_node
    AB = nodes[1] - nodes[0]
    AC = nodes[2] - nodes[0]
    area_vector = np.cross(AB, AC) / 2.0
    if np.dot(area_vector, ref_vect) < 0.0:
        area_vector = - area_vector
        return [area_vector, -1]
    return [area_vector, 1]
