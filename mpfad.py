import numpy as np
from pymoab import types
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import mpfad.helpers.geometric as geo
from math import pi


class MpfaD3D:
    def __init__(self, mesh_data):

        self.mesh_data = mesh_data
        self.mb = mesh_data.mb
        self.mtu = mesh_data.mtu

        self.dirichlet_tag = mesh_data.dirichlet_tag
        self.neumann_tag = mesh_data.neumann_tag
        self.perm_tag = mesh_data.perm_tag

        self.pressure_tag = self.mb.tag_get_handle(
            "Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        self.global_id_tag = self.mb.tag_get_handle(
            "GLOBAL_ID_VOLUME",
            1,
            types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE,
            True,
        )

        self.dirichlet_nodes = set(
            self.mb.get_entities_by_type_and_tag(
                0, types.MBVERTEX, self.dirichlet_tag, np.array((None,))
            )
        )

        self.neumann_nodes = set(
            self.mb.get_entities_by_type_and_tag(
                0, types.MBVERTEX, self.neumann_tag, np.array((None,))
            )
        )
        self.neumann_nodes = self.neumann_nodes - self.dirichlet_nodes

        boundary_nodes = self.dirichlet_nodes | self.neumann_nodes
        self.intern_nodes = set(mesh_data.all_nodes) - boundary_nodes

        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces

        self.all_faces = mesh_data.all_faces
        boundary_faces = self.dirichlet_faces | self.neumann_faces
        # print('ALL FACES', all_faces, len(all_faces))
        self.intern_faces = set(self.all_faces) - boundary_faces

        self.volumes = self.mesh_data.all_volumes

        self.A = lil_matrix(
            (len(self.volumes), len(self.volumes)), dtype=np.float
        )

        # self.A = np.zeros([len(self.volumes), len(self.volumes)])
        # print('CRIOU MATRIZ A')
        self.b = lil_matrix((len(self.volumes), 1), dtype=np.float)
        # self.b = np.zeros([1, len(self.volumes)])

    def _benchmark_1(self, x, y, z):
        K = [1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0]
        y = y + 1 / 2.0
        z = z + 1 / 3.0
        u1 = 1 + np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z)
        return K, u1

    def _area_vector(self, AB, AC, ref_vect=np.zeros(3), norma=False):
        # print('VECT', AB, AC)
        area_vector = np.cross(AB, AC) / 2.0
        if norma:
            area = np.sqrt(np.dot(area_vector, area_vector))
            return area
        if np.dot(area_vector, ref_vect) < 0.0:
            area_vector = -area_vector
            return [area_vector, -1]
        return [area_vector, 1]

    def area_vector(self, JK, JI, test_vector=np.zeros(3)):
        N_IJK = np.cross(JK, JI) / 2.0
        if test_vector.all() != (np.zeros(3)).all():
            check_left_or_right = np.dot(test_vector, N_IJK)
            if check_left_or_right < 0:
                N_IJK = -N_IJK
        return N_IJK

    def set_global_id(self):
        vol_ids = {}
        range_of_ids = range(len(self.mesh_data.all_volumes))
        for id_, volume in zip(range_of_ids, self.mesh_data.all_volumes):
            vol_ids[volume] = id_
        return vol_ids

    def _flux_term(self, vector_1st, permeab, vector_2nd, face_area=1.0):
        aux_1 = np.dot(vector_1st, permeab)
        aux_2 = np.dot(aux_1, vector_2nd)
        flux_term = aux_2 / face_area
        return flux_term

    # chamar essa função de cross_diffusion_term
    def _intern_cross_term(
        self,
        tan_vector,
        cent_vector,
        face_area,
        tan_term_1st,
        tan_term_2nd,
        norm_term_1st,
        norm_term_2nd,
        cent_dist_1st,
        cent_dist_2nd,
    ):
        mesh_aniso_term = np.dot(tan_vector, cent_vector) / (face_area ** 2.0)
        phys_aniso_term = ((tan_term_1st / norm_term_1st) * cent_dist_1st) - (
            (tan_term_2nd / norm_term_2nd) * cent_dist_2nd
        )

        cross_flux_term = mesh_aniso_term + phys_aniso_term / face_area
        return cross_flux_term

    def _boundary_cross_term(
        self,
        tan_vector,
        norm_vector,
        face_area,
        tan_flux_term,
        norm_flux_term,
        cent_dist,
    ):
        mesh_aniso_term = np.dot(tan_vector, norm_vector) / (face_area ** 2.0)
        phys_aniso_term = tan_flux_term / face_area
        cross_term = (
            mesh_aniso_term * (norm_flux_term / cent_dist) + phys_aniso_term
        )
        return cross_term

    def get_nodes_weights(self, method):
        self.nodes_ws = {}
        self.nodes_nts = {}
        for node in self.intern_nodes:
            self.nodes_ws[node] = method(node)
        for node in self.neumann_nodes:
            self.nodes_ws[node] = method(node, neumann=True)
            self.nodes_nts[node] = self.nodes_ws[node].pop(node)

    def _node_treatment(
        self,
        node,
        id_1st,
        id_2nd,
        v_ids,
        transm,
        cross_1st,
        cross_2nd=0.0,
        is_J=-1,
    ):
        value = (is_J) * transm * (cross_1st + cross_2nd)
        if node in self.dirichlet_nodes:
            x_n, y_n, z_n = self.mb.get_coords([node])
            pressure = self._benchmark_1(x_n, y_n, z_n)[1]
            print(pressure)
            self.b[id_1st, 0] += -value * pressure
            self.b[id_2nd, 0] += value * pressure

        if node in self.intern_nodes:
            x_n, y_n, z_n = self.mb.get_coords([node])
            pressure = self._benchmark_1(x_n, y_n, z_n)[1]
            print(pressure)

            self.b[id_1st, 0] += -value * pressure
            self.b[id_2nd, 0] += value * pressure

        if node in self.neumann_nodes:
            x_n, y_n, z_n = self.mb.get_coords([node])
            pressure = self._benchmark_1(x_n, y_n, z_n)[1]
            print(pressure)

            self.b[id_1st, 0] += -value * pressure
            self.b[id_2nd, 0] += value * pressure

    def run_solver(self):

        v_ids = self.set_global_id()
