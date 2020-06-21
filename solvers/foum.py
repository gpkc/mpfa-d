import numpy as np


class Foum:
    def __init__(
        self,
        mesh_data,
        water_specific_mass,
        oil_SG,
        viscosity_w,
        viscosity_o,
        cfl,
    ):
        self.mesh_data = mesh_data

        self.mb = mesh_data.mb
        self.mtu = mesh_data.mtu

        self.rho_w = water_specific_mass
        self.rho_o = oil_SG * water_specific_mass
        self.viscosity_w = viscosity_w
        self.viscosity_o = viscosity_o

        self.water_sat_i_tag = self.mesh_data.water_sat_i_tag
        self.oil_sat_i_tag = self.mesh_data.oil_sat_i_tag
        self.water_sat_bc_tag = self.mesh_data.water_sat_bc_tag
        self.rel_perm_w_tag = self.mesh_data.rel_perm_w_tag
        self.rel_perm_o_tag = self.mesh_data.rel_perm_o_tag
        self.water_sat_tag = self.mesh_data.water_sat_tag
        self.face_mobility_tag = self.mesh_data.face_mobility_tag
        self.source_tag = self.mesh_data.source_tag

        # self.face_velocity_tag = self.mesh_data.face_velocity_tag
        self.cfl = cfl

    def get_relative_perms(self, sat_W, sat_W_i, sat_O_i, nw=2, no=2):
        krw = ((sat_W - sat_W_i) / (1 - sat_W - sat_W_i - sat_O_i)) ** nw
        kro = ((1 - sat_W - sat_O_i) / (1 - sat_W_i - sat_O_i)) ** no
        return krw, kro

    def calc_mobility(self, kr, viscosity):
        return kr / viscosity

    def calc_face_mobility(self, lambda_l, v_l, lambda_r, v_r):
        return (lambda_l * v_l + lambda_r * v_r) / (v_l + v_r)

    def get_delta_t(self):
        pass

    def calc_fractional_flux(self, lambda_w, lambda_o):
        return lambda_w / (lambda_w + lambda_o)

    def init(self):
        s_w = np.asarray(
            self.mb.tag_get_data(
                self.water_sat_tag, self.mesh_data.all_volumes
            )
        ).flatten()
        sat_W_i = np.asarray(
            self.mb.tag_get_data(
                self.water_sat_i_tag, self.mesh_data.all_volumes
            )
        ).flatten()
        sat_O_i = np.asarray(
            self.mb.tag_get_data(
                self.oil_sat_i_tag, self.mesh_data.all_volumes
            )
        ).flatten()

        krw, kro = self.get_relative_perms(s_w, sat_W_i, sat_O_i)
        self.mb.tag_set_data(
            self.rel_perm_w_tag, self.mesh_data.all_volumes, krw
        )
        self.mb.tag_set_data(
            self.rel_perm_o_tag, self.mesh_data.all_volumes, kro
        )
        lambda_face = []
        for face in self.mesh_data.all_faces:
            try:
                vol_l, vol_r = self.mtu.get_bridge_adjacencies(face, 2, 3)

                nodes_l = self.mtu.get_bridge_adjacencies(vol_l, 3, 0)
                nodes_l_crds = self.mesh_data.mb.get_coords(nodes_l).reshape(
                    [4, 3]
                )
                vol_l_volume = self.mesh_data.get_tetra_volume(nodes_l_crds)

                krw_l = self.mb.tag_get_data(self.rel_perm_w_tag, vol_l)
                lambda_W_l = self.calc_mobility(krw_l, self.viscosity_w)
                kro_l = self.mb.tag_get_data(self.rel_perm_o_tag, vol_l)
                lambda_O_l = self.calc_mobility(kro_l, self.viscosity_o)

                nodes_r = self.mtu.get_bridge_adjacencies(vol_r, 3, 0)

                nodes_r_crds = self.mesh_data.mb.get_coords(nodes_r).reshape(
                    [4, 3]
                )
                vol_r_volume = self.mesh_data.get_tetra_volume(nodes_r_crds)

                krw_r = self.mb.tag_get_data(self.rel_perm_w_tag, vol_r)
                lambda_W_r = self.calc_mobility(krw_r, self.viscosity_w)
                kro_r = self.mb.tag_get_data(self.rel_perm_o_tag, vol_r)
                lambda_O_r = self.calc_mobility(kro_r, self.viscosity_o)

                lambda_face_W = self.calc_face_mobility(
                    lambda_W_l, vol_l_volume, lambda_W_r, vol_r_volume
                )
                lambda_face_O = self.calc_face_mobility(
                    lambda_O_l, vol_l_volume, lambda_O_r, vol_r_volume
                )
                lambda_face.append(lambda_face_W + lambda_face_O)

            except ValueError:
                vol_l = self.mtu.get_bridge_adjacencies(face, 2, 3)
                krw_l = self.mb.tag_get_data(self.rel_perm_w_tag, vol_l)[0]
                lambda_face_W = self.calc_mobility(krw_l, self.viscosity_w)
                kro_l = self.mb.tag_get_data(self.rel_perm_o_tag, vol_l)
                lambda_face_O = self.calc_mobility(kro_l, self.viscosity_o)
                lambda_face.append(lambda_face_W + lambda_face_O)
        lambda_face = np.asarray(lambda_face).flatten()
        self.mb.tag_set_data(
            self.face_mobility_tag, self.mesh_data.all_faces, lambda_face,
        )

    def run(self):
        raise NotImplementedError
