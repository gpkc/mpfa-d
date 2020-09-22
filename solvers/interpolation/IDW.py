import numpy as np

from .InterpolationMethod import InterpolationMethodBase
from solvers.helpers.geometric import point_distance


class IDW(InterpolationMethodBase):
    def calc_weight(self, node_coords, volume):
        vol_cent = self.mesh_data.get_centroid(volume)
        inv_dist = 1 / point_distance(node_coords, vol_cent)

        return inv_dist

    def interpolate(self, node, neumann=False):
        coords_node = self.mb.get_coords([node])
        vols_around = self.mtu.get_bridge_adjacencies(node, 0, 3)

        weights = np.array(
            [self.calc_weight(coords_node, vol) for vol in vols_around]
        )
        weights = weights / np.sum(weights)

        node_weights = dict(zip(vols_around, weights))
        if neumann:
            node_weights[node] = 0.0
        # print(node_weights)
        return node_weights
