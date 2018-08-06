import numpy as np
from pymoab import types
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import mpfad.helpers.geometric as geo


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

        self.A = lil_matrix((len(self.volumes), len(self.volumes)),
                            dtype=np.float)

        # self.A = np.zeros([len(self.volumes), len(self.volumes)])
        # print('CRIOU MATRIZ A')
        self.b = lil_matrix((len(self.volumes), 1),
                            dtype=np.float)
        # self.b = np.zeros([1, len(self.volumes)])

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

    # chamar essa função de cross_diffusion_term
    def _intern_cross_term(self, tan_vector, cent_vector, face_area,
                           tan_term_1st, tan_term_2nd,
                           norm_term_1st, norm_term_2nd,
                           cent_dist_1st, cent_dist_2nd):
        mesh_aniso_term = np.dot(tan_vector, cent_vector) / (face_area ** 2.0)
        phys_aniso_term = ((tan_term_1st / norm_term_1st) * cent_dist_1st) + \
                          ((tan_term_2nd / norm_term_2nd) * cent_dist_2nd)

        cross_flux_term = mesh_aniso_term - phys_aniso_term / face_area
        return cross_flux_term

    def _boundary_cross_term(self, tan_vector, norm_vector, face_area,
                             tan_flux_term, norm_flux_term, cent_dist):
        mesh_aniso_term = np.dot(tan_vector, norm_vector)/(face_area ** 2.0)
        phys_aniso_term = tan_flux_term / face_area
        cross_term = mesh_aniso_term * (norm_flux_term / cent_dist) - \
            phys_aniso_term
        return cross_term

    def get_nodes_weights(self, method):
        self.nodes_ws = {}
        self.nodes_nts = {}
        for node in self.intern_nodes:
            self.nodes_ws[node] = method(node)
        for node in self.neumann_nodes:
            self.nodes_ws[node] = method(node, neumann=True)
            self.nodes_nts[node] = self.nodes_ws[node].pop(node)

    def _node_treatment(self, node, id_1st, id_2nd, v_ids,
                        transm, cross_1st, cross_2nd=0.0, is_J=1):
        value = (is_J) * transm * (cross_1st + cross_2nd)
        if node in self.dirichlet_nodes:
            pressure = self.mesh_data.mb.tag_get_data(self.dirichlet_tag, node)

            self.b[id_1st, 0] += - value * pressure
            self.b[id_2nd, 0] += value * pressure

        if node in self.intern_nodes:
            for volume, weight in self.nodes_ws[node].items():
                v_id = v_ids[volume]

                self.A[id_1st, v_id] += value * weight
                self.A[id_2nd, v_id] += - value * weight

        if node in self.neumann_nodes:
            neu_term = self.nodes_nts[node]

            self.b[id_1st, 0] += - value * neu_term
            self.b[id_2nd, 0] += value * neu_term

            for volume, weight_N in self.nodes_ws[node].items():
                v_id = v_ids[volume]

                self.A[id_1st, v_id] += value * weight_N
                self.A[id_2nd, v_id] += - value * weight_N

    def __node_treatment(self, node, id_1st, id_2nd, v_ids, RHS):
        value = RHS
        if node in self.dirichlet_nodes:
            pressure = self.mesh_data.mb.tag_get_data(self.dirichlet_tag, node)
            # print('DIR', value, pressure)

            self.b[id_1st, 0] += - value * pressure
            self.b[id_2nd, 0] += value * pressure

        if node in self.intern_nodes:
            for volume, weight in self.nodes_ws[node].items():
                v_id = v_ids[volume]

                self.A[id_1st, v_id] += value * weight
                self.A[id_2nd, v_id] += - value * weight

        if node in self.neumann_nodes:
            neu_term = self.nodes_nts[node]

            self.b[id_1st, 0] += - value * neu_term
            self.b[id_2nd, 0] += value * neu_term

            for volume, weight_N in self.nodes_ws[node].items():
                v_id = v_ids[volume]
                # print('VOL ', v_id, weight_N)

                self.A[id_1st, v_id] += value * weight_N
                self.A[id_2nd, v_id] += - value * weight_N

    def run_solver(self, interpolation_method):

        self.get_nodes_weights(interpolation_method)
        v_ids = self.set_global_id()

        for face in self.all_faces:

            if face in self.neumann_faces:
                face_flow = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
                volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                volume = np.asarray(volume, dtype='uint64')
                face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
                node_crds = self.mb.get_coords(face_nodes).reshape([3, 3])
                # print("BBBB", self.b, volume[0], v_ids[volume[0]], self.b[10,0])
                face_area = geo._area_vector(node_crds, norma=True)
                # print('FACE AREA', face_area, face_flow)
                self.b[v_ids[volume[0]], 0] += - face_flow * face_area

            if face in self.dirichlet_faces:
                volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                volume = np.asarray(volume, dtype='uint64')
                R = self.mtu.get_average_position(volume)
                face_centroid = self.mtu.get_average_position([face])
                nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
                I, J, K = nodes
                g_I = self.mb.tag_get_data(self.dirichlet_tag, I)
                g_J = self.mb.tag_get_data(self.dirichlet_tag, J)
                g_K = self.mb.tag_get_data(self.dirichlet_tag, K)

                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                test_vector = face_centroid - R
                # print('IN VECTS ', JK, JI)
                _eval = self._area_vector(JK, JI, test_vector)
                N_IJK = _eval[0]
                test = _eval[1]

                if test == -1:
                    K, I = I, K
                    JK, JI = JI, JK
                nodes = np.asarray([I, J, K], dtype='uint64')
                nodes_crds = self.mb.get_coords(nodes).reshape([len(nodes), 3])
                #####
                JR = R - self.mb.get_coords([J])
                h_R = np.absolute(np.dot(N_IJK, JR) / np.sqrt(np.dot(N_IJK,
                                                              N_IJK)))
                K_R = self.mb.tag_get_data(self.perm_tag,
                                           volume).reshape([3, 3])
                K_R_n = self._flux_term(N_IJK, K_R, N_IJK, face_area)
                K_n_eff = K_R_n / h_R

                face_area = np.sqrt(np.dot(N_IJK, N_IJK))
                RHS = 0.0
                for i in range(len(nodes)):
                    p_i = self.mb.tag_get_data(self.dirichlet_tag, nodes[i])
                    op_nodes = np.array([nodes_crds[i-1], nodes_crds[i-2], R])
                    N_i = geo._area_vector(op_nodes, nodes_crds[i])[0]
                    # print('TEST ', N_IJK, K_R, N_i, face_area)
                    K_R_i = self._flux_term(N_IJK, K_R, N_i, face_area)
                    RHS += p_i * K_R_i

                RHS = RHS[0][0] / h_R
                # print('RHS', RHS)

                volume_id = v_ids[volume[0]]
                self.A[volume_id, volume_id] += K_n_eff
                self.b[volume_id, 0] += - RHS
                #####
                """
                JR = R - self.mb.get_coords([J])
                tan_JI = np.cross(JI, N_IJK)
                # if np.dot(tan_JI, JK) < 0.0:
                #     tan_JI = -tan_JI
                tan_JK = np.cross(N_IJK, JK)
                # if np.dot(tan_JK, JI) < 0.0:
                #     tan_JK = -tan_JK

                h_R = np.absolute(np.dot(N_IJK, JR) / np.sqrt(np.dot(N_IJK,
                                                              N_IJK)))
                face_area = np.sqrt(np.dot(N_IJK, N_IJK))

                K_R = self.mb.tag_get_data(self.perm_tag,
                                           volume).reshape([3, 3])
                K_R_n = self._flux_term(N_IJK, K_R, N_IJK, face_area)
                K_R_JI = self._flux_term(N_IJK, K_R, tan_JI, face_area)
                K_R_JK = self._flux_term(N_IJK, K_R, tan_JK, face_area)

                D_JI = self._boundary_cross_term(-tan_JK, JR, face_area,
                                                 K_R_JK, K_R_n, h_R)
                D_JK = self._boundary_cross_term(-tan_JI, JR, face_area,
                                                 K_R_JI, K_R_n, h_R)
                K_n_eff = K_R_n / h_R

                # RHS = -(D_JI * (g_J - g_I) + D_JK * (g_J - g_K) +
                #         2 * K_R_n / h_R * g_J)

                RHS = ((2.0 * K_R_n / h_R) - D_JI - D_JK) * g_J + D_JI * g_I + D_JK * g_K
                volume_id = v_ids[volume[0]]

                self.A[volume_id, volume_id] += -2.0 * K_n_eff
                self.b[volume_id, 0] += - RHS
                """

            if face in self.intern_faces:
                face_centroid = self.mtu.get_average_position([face])
                left_volume, right_volume = \
                    self.mtu.get_bridge_adjacencies(face, 2, 3)
                L = self.mtu.get_average_position([left_volume])
                R = self.mtu.get_average_position([right_volume])

                nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
                I, J, K = nodes
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])

                RJ = self.mb.get_coords([J]) - R

                _eval = self._area_vector(JK, JI, RJ)
                N_IJK = _eval[0]
                test = _eval[1]

                if test == -1:
                    # print('TEST', I, K, JK, JI)
                    K, I = I, K
                    JK, JI = JI, JK
                    # print('AFTER', I, K, JK, JI)

                # print('PRO', N_IJK, nodes_crds, R, RJ, R, J)
                nodes = np.asarray([I, J, K], dtype='uint64')
                nodes_crds = self.mb.get_coords(nodes).reshape([len(nodes), 3])
                face_area = np.sqrt(np.dot(N_IJK, N_IJK))

                ####
                K_R = self.mb.tag_get_data(self.perm_tag,
                                           right_volume).reshape([3, 3])
                h_R = face_centroid - R
                h_R = np.absolute(np.dot(N_IJK, h_R) / np.sqrt(np.dot(N_IJK,
                                                                      N_IJK)))
                K_R_n = self._flux_term(N_IJK, K_R, N_IJK, face_area)

                K_L = self.mb.tag_get_data(self.perm_tag,
                                           left_volume).reshape([3, 3])
                h_L = face_centroid - L
                h_L = np.absolute(np.dot(N_IJK, h_L) / np.sqrt(np.dot(N_IJK,
                                                                      N_IJK)))
                K_L_n = self._flux_term(N_IJK, K_L, N_IJK, face_area)

                den_K = K_R_n * h_L + K_L_n * h_R
                K_n_eff = K_R_n * K_L_n / den_K

                gid_right = v_ids[right_volume]
                gid_left = v_ids[left_volume]

                self.A[gid_right, gid_right] += K_n_eff
                self.A[gid_right, gid_left] += - K_n_eff

                self.A[gid_left, gid_left] += K_n_eff
                self.A[gid_left, gid_right] += - K_n_eff

                for i in range(len(nodes)):
                    op_crds_R = np.array([nodes_crds[i-1], nodes_crds[i-2], R])
                    op_crds_L = np.array([nodes_crds[i-1], nodes_crds[i-2], L])

                    N_Ri = geo._area_vector(op_crds_R, nodes_crds[i])[0]
                    N_Li = geo._area_vector(op_crds_L, nodes_crds[i])[0]

                    # print('TEST ', N_IJK, K_R, N_i, face_area)
                    K_R_i = self._flux_term(N_IJK, K_R, N_Ri, face_area)
                    K_L_i = self._flux_term(- N_IJK, K_L, N_Li, face_area)

                    RHS = (K_R_i * K_L_n - K_L_i * K_R_n) / den_K
                    # print('NOD', nodes_crds[i], nodes_crds[i-1], nodes_crds[i-2], R, L)
                    # print('PRO', N_IJK, nodes_crds, R)
                    # print('VEC', N_IJK, N_Ri, N_Li)
                    # print('RHS', RHS, K_R_i, K_L_n, K_L_i, K_R_n, den_K)

                    self.__node_treatment(nodes[i], gid_right, gid_left,
                                          v_ids, RHS)
                ####
                """
                dist_LR = R - L

                tan_JI = np.cross(JI, N_IJK)
                # if np.dot(tan_JI, JK) < 0.0:
                #     tan_JI = -tan_JI

                tan_JK = np.cross(N_IJK, JK)
                # if np.dot(tan_JK, JI) < 0.0:
                #     tan_JK = -tan_JK

                K_R = self.mb.tag_get_data(self.perm_tag,
                                           right_volume).reshape([3, 3])

                h_R = face_centroid - R
                h_R = np.absolute(np.dot(N_IJK, h_R) / np.sqrt(np.dot(N_IJK,
                                                                      N_IJK)))

                K_R_n = self._flux_term(N_IJK, K_R, N_IJK, face_area)
                K_R_JI = self._flux_term(N_IJK, K_R, tan_JI, face_area)
                K_R_JK = self._flux_term(N_IJK, K_R, tan_JK, face_area)

                K_L = self.mb.tag_get_data(self.perm_tag,
                                           left_volume).reshape([3, 3])
                h_L = face_centroid - L
                h_L = np.absolute(np.dot(N_IJK, h_L) / np.sqrt(np.dot(N_IJK,
                                                                      N_IJK)))

                K_L_n = self._flux_term(N_IJK, K_L, N_IJK, face_area)
                K_L_JI = self._flux_term(N_IJK, K_L, tan_JI, face_area)
                K_L_JK = self._flux_term(N_IJK, K_L, tan_JK, face_area)

                D_JI = self._intern_cross_term(-tan_JK, dist_LR, face_area,
                                               K_R_JK, K_L_JK,
                                               K_R_n, K_L_n,
                                               h_R, h_L)

                D_JK = self._intern_cross_term(-tan_JI, dist_LR, face_area,
                                               K_R_JI, K_L_JI,
                                               K_R_n, K_L_n,
                                               h_R, h_L)

                K_n_eff = K_R_n * K_L_n / (K_R_n * h_L + K_L_n * h_R)

                gid_right = v_ids[right_volume]
                gid_left = v_ids[left_volume]

                self.A[gid_right, gid_right] += -2.0 * K_n_eff
                self.A[gid_right, gid_left] += 2.0 * K_n_eff

                self.A[gid_left, gid_left] += -2.0 * K_n_eff
                self.A[gid_left, gid_right] += 2.0 * K_n_eff

                self._node_treatment(I, gid_right, gid_left,
                                     v_ids, K_n_eff, D_JI)
                self._node_treatment(K, gid_right, gid_left,
                                     v_ids, K_n_eff, D_JK)
                self._node_treatment(J, gid_right, gid_left,
                                     v_ids, K_n_eff, D_JI, cross_2nd=D_JK,
                                     is_J=-1)

        """
        print(' ')
        # print(self.A.todense())
        print('\n'.join([''.join(['{:10.4f}'.format(item) for item in row]) for row in np.asarray(self.A.todense())]))
        print(self.b.todense())
        # for vol in self.volumes:
        #     print(v_ids[vol], self.mtu.get_average_position([vol]))

        self.A = self.A.tocsc()
        self.b = self.b.tocsc()

        p = spsolve(self.A, self.b)
        # print('Pressure', p)
        # p = np.linalg.solve(self.A, self.b[0])
        self.mb.tag_set_data(self.pressure_tag, self.volumes, p)

    def record_data(self, file_name):
        volumes = self.mb.get_entities_by_dimension(0, 3)
        ms = self.mb.create_meshset()
        self.mb.add_entities(ms, volumes)
        self.mb.write_file(file_name, [ms])
