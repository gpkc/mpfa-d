from math import sqrt

import numpy as np


def point_distance(coords_1, coords_2):
    dist_vector = coords_1 - coords_2
    distance = sqrt(np.dot(dist_vector, dist_vector))
    return distance
