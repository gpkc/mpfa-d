import numpy as np
import time
from pymoab import types
from scipy.sparse import lil_matrix, csr_matrix
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
            "Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.global_id_tag = self.mb.tag_get_handle(
                        "GLOBAL_ID_VOLUME", 1, types.MB_TYPE_DOUBLE,
                        types.MB_TAG_SPARSE, True)

        self.dirichlet_nodes = set(self.mb.get_entities_by_type_and_tag(
            0, types.MBVERTEX, self.dirichlet_tag, np.array((None,))))

        self.neumann_nodes = set(self.mb.get_entities_by_type_and_tag(
            0, types.MBVERTEX, self.neumann_tag, np.array((None,))))
        self.neumann_nodes = self.neumann_nodes - self.dirichlet_nodes

        boundary_nodes = (self.dirichlet_nodes | self.neumann_nodes)
        self.intern_nodes = set(mesh_data.all_nodes) - boundary_nodes

        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces

        self.all_faces = mesh_data.all_faces
        boundary_faces = (self.dirichlet_faces | self.neumann_faces)
        # print('ALL FACES', all_faces, len(all_faces))
        self.intern_faces = set(self.all_faces) - boundary_faces

        self.volumes = self.mesh_data.all_volumes

        self.A = csr_matrix((len(self.volumes), len(self.volumes)),
                            dtype=np.float)
        self.b = csr_matrix((len(self.volumes), 1),
                            dtype=np.float)

    def _benchmark_1(self, x, y, z):
        K = [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
        y = y + 1/2.
        z = z + 1/3.
        x = x + 0.
        u1 = 1 + np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z)
        return K, x ** 3  # u1

    def calculate_gradient(self, x, y, z, benchmark, delta=0.00001):
        perm = np.array(benchmark(x, y, z)[0]).reshape([3, 3])
        grad_x = (benchmark(x + delta, y, z)[1] -
                  benchmark(x, y, z)[1]) / delta
        grad_y = (benchmark(x, y + delta, z)[1] -
                  benchmark(x, y, z)[1]) / delta
        grad_z = (benchmark(x, y, z + delta)[1] -
                  benchmark(x, y, z)[1]) / delta
        grad = np.array([grad_x, grad_y, grad_z])
        return perm * grad

    def calculate_divergent(self, x, y, z, benchmark, delta=0.00001):
        k_grad_x = (self.calculate_gradient(x + delta, y, z, benchmark)[0]
                    - self.calculate_gradient(x, y, z, benchmark)[0]) / delta
        k_grad_y = (self.calculate_gradient(x, y + delta, z, benchmark)[1]
                    - self.calculate_gradient(x, y, z, benchmark)[1]) / delta
        k_grad_z = (self.calculate_gradient(x, y, z + delta, benchmark)[2]
                    - self.calculate_gradient(x, y, z, benchmark)[2]) / delta
        return -np.sum(k_grad_x + k_grad_y + k_grad_z)

    def _area_vector(self, AB, AC, ref_vect=np.zeros(3), norma=False):
        # print('VECT', AB, AC)
        area_vector = np.cross(AB, AC) / 2.0
        if norma:
            area = np.sqrt(np.dot(area_vector, area_vector))
            return area
        if np.dot(area_vector, ref_vect) < 0.0:
            area_vector = - area_vector
            return [area_vector, -1]
        return [area_vector, 1]

    def area_vector(self, JK, JI, test_vector=np.zeros(3)):
        N_IJK = np.cross(JK, JI) / 2.0
        if test_vector.all() != (np.zeros(3)).all():
            check_left_or_right = np.dot(test_vector, N_IJK)
            if check_left_or_right < 0:
                N_IJK = - N_IJK
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

    def _intern_cross_term(self, tan_vector, cent_vector, face_area,
                           tan_term_1st, tan_term_2nd,
                           norm_term_1st, norm_term_2nd,
                           cent_dist_1st, cent_dist_2nd):
        mesh_aniso_term = np.dot(tan_vector, cent_vector) / (2 * face_area
                                                             ** 2.0)
        phys_aniso_term = ((tan_term_1st / norm_term_1st) * cent_dist_1st) - \
                          ((tan_term_2nd / norm_term_2nd) * cent_dist_2nd)

        cross_flux_term = mesh_aniso_term + phys_aniso_term / (2 * face_area)
        return cross_flux_term

    def _boundary_cross_term(self, tan_vector, norm_vector, face_area,
                             tan_flux_term, norm_flux_term, cent_dist):
        mesh_aniso_term = np.dot(tan_vector, norm_vector)/(2 * face_area
                                                           ** 2.0)
        phys_aniso_term = tan_flux_term / (2 * face_area)
        cross_term = mesh_aniso_term * (norm_flux_term / cent_dist) + \
            phys_aniso_term
        return cross_term

    def get_nodes_weights(self, method):
        self.nodes_ws = {}
        self.nodes_nts = {}
        count = 0
        for node in self.intern_nodes:
            print(f"{count} / {len(self.intern_nodes)}")
            count += 1
            self.nodes_ws[node] = method(node)
        for node in self.neumann_nodes:
            self.nodes_ws[node] = method(node, neumann=True)
            self.nodes_nts[node] = self.nodes_ws[node].pop(node)

    def _node_treatment(self, node, id_1st, id_2nd, v_ids,
                        transm, cross_1st, cross_2nd=0.0, is_J=1):
        value = (is_J) * transm * (cross_1st + cross_2nd)
        if node in self.dirichlet_nodes:
            # x_n, y_n, z_n = self.mb.get_coords([node])
            # pressure = self._benchmark_1(x_n, y_n, z_n)[1]
            pressure = self.mesh_data.mb.tag_get_data(self.dirichlet_tag, node)

            self.b[id_1st, 0] += -value * pressure
            self.b[id_2nd, 0] += value * pressure

        if node in self.intern_nodes:
            # x_n, y_n, z_n = self.mb.get_coords([node])
            # pressure = self._benchmark_1(x_n, y_n, z_n)[1]
            # self.b[id_1st, 0] += -value * pressure
            # self.b[id_2nd, 0] += value * pressure
            for volume, weight in self.nodes_ws[node].items():
                v_id = v_ids[volume]

                self.A[id_1st, v_id] += value * weight
                self.A[id_2nd, v_id] += -value * weight

        if node in self.neumann_nodes:
            # x_n, y_n, z_n = self.mb.get_coords([node])
            # pressure = self._benchmark_1(x_n, y_n, z_n)[1]
            # self.b[id_1st, 0] += -value * pressure
            # self.b[id_2nd, 0] += value * pressure
            neu_term = self.nodes_nts[node]

            self.b[id_1st, 0] += -value * neu_term
            self.b[id_2nd, 0] += value * neu_term

            for volume, weight_N in self.nodes_ws[node].items():
                v_id = v_ids[volume]

                self.A[id_1st, v_id] += value * weight_N
                self.A[id_2nd, v_id] += -value * weight_N

    def run_solver(self, interpolation_method):
        begin = time.time()

        print("interpolating...")
        t0 = time.time()
        self.get_nodes_weights(interpolation_method)
        delta_t_interpol = time.time() - t0
        print("took {0} seconds to interpolate...".format(delta_t_interpol))

        v_ids = self.set_global_id()

        print('filling b vector with source term...')
        t0 = time.time()
        for volume in self.volumes:
            volume_id = v_ids[volume]
            x, y, z = self.mtu.get_average_position([volume])
            vol_nodes = self.mb.get_adjacencies(volume, 0)
            vol_nodes_crds = self.mb.get_coords(vol_nodes)
            vol_nodes_crds = np.reshape(vol_nodes_crds, (4, 3))
            tetra_vol = self.mesh_data.get_tetra_volume(vol_nodes_crds)
            source_term = self.calculate_divergent(x, y, z, self._benchmark_1)
            self.b[volume_id, 0] += source_term * tetra_vol
        delta_t_source_term = time.time() - t0
        print("took {0} seconds to fill source term...".format(
              delta_t_source_term))

        print('filling Matrix....')
        t0 = time.time()

        for face in self.all_faces:

            if face in self.neumann_faces:
                face_flow = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
                volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                volume = np.asarray(volume, dtype='uint64')
                face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
                node_crds = self.mb.get_coords(face_nodes).reshape([3, 3])
                face_area = geo._area_vector(node_crds, norma=True)
                self.b[v_ids[volume[0]], 0] += - face_flow * face_area

            if face in self.dirichlet_faces:
                volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                volume = np.asarray(volume, dtype='uint64')
                volume_centroid = self.mtu.get_average_position(volume)
                face_centroid = self.mtu.get_average_position([face])
                I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)

                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                test_vector = face_centroid - volume_centroid
                N_IJK, test = self._area_vector(JK, JI, test_vector)

                if test == -1:
                    K, I = I, K
                    JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                    JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                    N_IJK = self._area_vector(JK, JI, test_vector)[0]

                g_I = self.mb.tag_get_data(self.dirichlet_tag, I)
                g_J = self.mb.tag_get_data(self.dirichlet_tag, J)
                g_K = self.mb.tag_get_data(self.dirichlet_tag, K)

                JR = volume_centroid - self.mb.get_coords([J])
                tan_JI = np.cross(N_IJK, JI)
                tan_JK = np.cross(JK, N_IJK)

                h_R = np.absolute(np.dot(N_IJK, JR) / np.sqrt(np.dot(N_IJK,
                                                              N_IJK)))
                face_area = np.sqrt(np.dot(N_IJK, N_IJK))

                K_R = self.mb.tag_get_data(self.perm_tag,
                                           volume).reshape([3, 3])
                K_R_n = self._flux_term(N_IJK, K_R, N_IJK, face_area)
                K_R_JI = self._flux_term(N_IJK, K_R, tan_JI, face_area)
                K_R_JK = self._flux_term(N_IJK, K_R, tan_JK, face_area)

                D_JI = self._boundary_cross_term(tan_JK, JR, face_area,
                                                 K_R_JK, K_R_n, h_R)
                D_JK = self._boundary_cross_term(tan_JI, JR, face_area,
                                                 K_R_JI, K_R_n, h_R)

                K_n_eff = K_R_n / h_R
                RHS = -(D_JI * (g_J - g_I) + D_JK * (g_J - g_K) +
                        K_R_n / h_R * g_J)

                volume_id = v_ids[volume[0]]
                self.A[volume_id, volume_id] += K_n_eff
                self.b[volume_id, 0] += - RHS

            if face in self.intern_faces:
                face_centroid = self.mtu.get_average_position([face])
                left_volume, right_volume = \
                    self.mtu.get_bridge_adjacencies(face, 2, 3)
                L = self.mtu.get_average_position([left_volume])
                R = self.mtu.get_average_position([right_volume])

                I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])

                RJ = J - R

                N_IJK, test = self._area_vector(JK, JI, RJ)

                if test == -1:
                    K, I = I, K
                    JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                    JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                    N_IJK = self._area_vector(JK, JI, RJ)[0]

                face_area = np.sqrt(np.dot(N_IJK, N_IJK))

                dist_LR = R - L
                tan_JI = np.cross(N_IJK, JI)
                tan_JK = np.cross(JK, N_IJK)

                h_R = face_centroid - R
                h_R = np.absolute(np.dot(N_IJK, h_R) / np.sqrt(np.dot(N_IJK,
                                                                      N_IJK)))
                K_R = self.mb.tag_get_data(self.perm_tag,
                                           right_volume).reshape([3, 3])
                K_R_n = self._flux_term(N_IJK, K_R, N_IJK, face_area)
                K_R_JI = self._flux_term(N_IJK, K_R, tan_JI, face_area)
                K_R_JK = self._flux_term(N_IJK, K_R, tan_JK, face_area)

                h_L = face_centroid - L
                h_L = np.absolute(np.dot(N_IJK, h_L) / np.sqrt(np.dot(N_IJK,
                                                                      N_IJK)))

                K_L = self.mb.tag_get_data(self.perm_tag,
                                           left_volume).reshape([3, 3])
                K_L_n = self._flux_term(N_IJK, K_L, N_IJK, face_area)
                K_L_JI = self._flux_term(N_IJK, K_L, tan_JI, face_area)
                K_L_JK = self._flux_term(N_IJK, K_L, tan_JK, face_area)

                D_JI = self._intern_cross_term(tan_JK, dist_LR, face_area,
                                               K_R_JK, K_L_JK,
                                               K_R_n, K_L_n,
                                               h_R, h_L)

                D_JK = self._intern_cross_term(tan_JI, dist_LR, face_area,
                                               K_R_JI, K_L_JI,
                                               K_R_n, K_L_n,
                                               h_R, h_L)

                K_n_eff = K_R_n * K_L_n / (K_R_n * h_L + K_L_n * h_R)

                gid_right = v_ids[right_volume]
                gid_left = v_ids[left_volume]

                self.A[gid_right, gid_right] += K_n_eff
                self.A[gid_right, gid_left] += - K_n_eff

                self.A[gid_left, gid_left] += K_n_eff
                self.A[gid_left, gid_right] += - K_n_eff

                self._node_treatment(I, gid_right, gid_left,
                                     v_ids, K_n_eff, D_JI)
                self._node_treatment(K, gid_right, gid_left,
                                     v_ids, K_n_eff, D_JK)
                self._node_treatment(J, gid_right, gid_left,
                                     v_ids, K_n_eff, D_JI, cross_2nd=D_JK,
                                     is_J=-1)

        self.A = self.A.tocsc()
        self.b = self.b.tocsc()
        delta_t_matrix = time.time() - t0
        print('complete! Took {0} to set the problem'.format(delta_t_matrix))
        t0 = time.time()
        p = spsolve(self.A, self.b)
        delta_t_solution = time.time() - t0
        print('solving problem took {0} seconds...'.format(delta_t_solution))
        self.mb.tag_set_data(self.pressure_tag, self.volumes, p)
        end = time.time() - begin
        print('took {0} seconds to run problem'.format(end))

    def record_data(self, file_name):
        volumes = self.mb.get_entities_by_dimension(0, 3)
        ms = self.mb.create_meshset()
        self.mb.add_entities(ms, volumes)
        self.mb.write_file(file_name, [ms])
