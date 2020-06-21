from .InterpolationMethod import InterpolationMethodBase


class LSW(InterpolationMethodBase):
    def calc_G(self, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz):
        G = (
            I_xx * (I_yy * I_zz - I_yz * I_yz)
            + I_xy * (I_yz * I_xz - I_xy * I_zz)
            + I_xz * (I_xy * I_yz - I_yy * I_xz)
        )

        return G

    def calc_psi(self, R_i, R_j, R_k, I_ij, I_ik, I_jj, I_jk, I_kk, G):
        psi_i = (
            R_i * (I_jk * I_jk - I_jj * I_kk)
            + R_j * (I_ij * I_kk - I_jk * I_ik)
            + R_k * (I_jj * I_ik - I_ij * I_jk)
        ) / G
        return psi_i

    def interpolate(self, node, neumann=False):
        crds_node = self.mb.get_coords([node])
        vols_around = self.mtu.get_bridge_adjacencies(node, 0, 3)

        R_x, R_y, R_z = 0.0, 0.0, 0.0
        I_xx, I_yy, I_zz = 0.0, 0.0, 0.0
        I_xy, I_xz, I_yz = 0.0, 0.0, 0.0

        rel_vol_position = []
        for a_vol in vols_around:
            vol_cent = self.mesh_data.get_centroid(a_vol)
            x_k, y_k, z_k = vol_cent - crds_node
            rel_vol_position.append((x_k, y_k, z_k, a_vol))

            R_x += x_k
            R_y += y_k
            R_z += z_k
            I_xx += x_k * x_k
            I_yy += y_k * y_k
            I_zz += z_k * z_k
            I_xy += x_k * y_k
            I_xz += x_k * z_k
            I_yz += y_k * z_k

        G = (
            I_xx * (I_yy * I_zz - I_yz * I_yz)
            + I_xy * (I_yz * I_xz - I_xy * I_zz)
            + I_xz * (I_xy * I_yz - I_yy * I_xz)
        )

        psi_x = (
            R_x * (I_yz * I_yz - I_yy * I_zz)
            + R_y * (I_xy * I_zz - I_yz * I_xz)
            + R_z * (I_yy * I_xz - I_xy * I_yz)
        ) / G

        psi_y = (
            R_x * (I_xy * I_zz - I_yz * I_xz)
            + R_y * (I_xz * I_xz - I_xx * I_zz)
            + R_z * (I_xx * I_yz - I_xy * I_xz)
        ) / G

        psi_z = (
            R_x * (I_yy * I_xz - I_xy * I_yz)
            + R_y * (I_xx * I_yz - I_xy * I_xz)
            + R_z * (I_xy * I_xy - I_xx * I_yy)
        ) / G

        num_vols = len(vols_around)
        nodes_weights = {}

        for x_k, y_k, z_k, volume in rel_vol_position:
            numerator = 1.0 + x_k * psi_x + y_k * psi_y + z_k * psi_z
            denominator = num_vols + R_x * psi_x + R_y * psi_y + R_z * psi_z
            nodes_weights[volume] = numerator / denominator

        if neumann:
            nodes_weights[node] = 0.0
        return nodes_weights
