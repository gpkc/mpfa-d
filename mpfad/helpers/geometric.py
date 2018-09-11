from pymoab import core
from pymoab import topo_util
import numpy as np

mb = core.Core()
root_set = mb.get_root_set()
mtu = topo_util.MeshTopoUtil(mb)

def point_distance(coords_1, coords_2):
    dist_vector = coords_1 - coords_2
    distance = np.sqrt(np.dot(dist_vector, dist_vector))
    return distance

def _area_vector(nodes, ref_node=np.zeros(3), norma=False):
    ref_vect = nodes[0] - ref_node
    AB = nodes[1] - nodes[0]
    AC = nodes[2] - nodes[0]
    area_vector = np.cross(AB, AC) / 2.0
    if norma:
        area = np.sqrt(np.dot(area_vector, area_vector))
        return area
    if np.dot(area_vector, ref_vect) < 0.0:
        area_vector = - area_vector
        return [area_vector, -1]
    return [area_vector, 1]

def get_height(normal_vector, distance_vector):
    return np.absolute(np.dot(normal_vector, distance_vector) /
            np.sqrt(np.dot(normal_vector, normal_vector)))

def get_tetra_volume(verts):
    pass
