import numpy as np
from pymoab import types


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
        self.A = np.zeros([len(self.volumes), len(self.volumes)])
        self.b = np.zeros([1, len(self.volumes)])

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

    def _flux_term(self, vector_1st, permeab, vector_2nd, face_area):
        aux_1 = np.dot(vector_1st, permeab)
        aux_2 = np.dot(aux_1, vector_2nd)
        flux_term = aux_2 / face_area
        return flux_term

    def _intern_cross_term(self, tan_vector, cent_vector, face_area,
                           tan_term_1st, tan_term_2nd,
                           norm_term_1st, norm_term_2nd,
                           cent_dist_1st, cent_dist_2nd):
        mesh_aniso_term = np.dot(tan_vector, cent_vector) / (face_area**2.0)
        phys_aniso_term = ((tan_term_1st / norm_term_1st) * cent_dist_1st) + \
                           ((tan_term_2nd / norm_term_2nd) * cent_dist_2nd)

        cross_flux_term = mesh_aniso_term - phys_aniso_term / face_area
        return cross_flux_term

    def _boundary_cross_term(self, tan_vector, norm_vector, face_area,
                             tan_flux_term, norm_flux_term, cent_dist):
        mesh_aniso_term = np.dot(tan_vector, norm_vector)/(face_area**2)
        phys_aniso_term = tan_flux_term / face_area
        cross_term = mesh_aniso_term * (norm_flux_term / cent_dist) - \
            phys_aniso_term
        return cross_term

    def get_intern_nodes_weights(self, method):
        nodes_weights = {}
        for a_node in self.intern_nodes:
            nodes_weights[a_node] = method(a_node)
        return nodes_weights

    def _node_treatment(self, node, nodes_weights, id_1st, id_2nd, v_ids,
                        transm, cross_1st, cross_2nd=0.0, is_J=1):
        if node in self.dirichlet_nodes:
            pressure = self.mesh_data.mb.tag_get_data(self.dirichlet_tag, node)
            value = (is_J) * transm * (cross_1st + cross_2nd) * pressure
            self.b[0][id_1st] += value
            self.b[0][id_2nd] += - value
            # print('VALOR', value)
        if node in self.intern_nodes:
            for volume, weight in nodes_weights[node].items():
                v_id = v_ids[volume]
                value = (is_J) * transm * (cross_1st + cross_2nd) * weight
                self.A[id_1st][v_id] += - value
                self.A[id_2nd][v_id] += value

    def run_solver(self, interpolation_method):

        nodes_weights = self.get_intern_nodes_weights(interpolation_method)

        v_ids = self.set_global_id()

        for a_node in self.neumann_nodes:  # | self.intern_nodes:
            a_node_coords = self.mb.get_coords([a_node])[0]
            self.mb.tag_set_data(self.dirichlet_tag, a_node, 1.0 - a_node_coords)

        for face in self.all_faces:

            if face in self.neumann_faces:
                face_flow = self.mb.tag_get_data(self.neumann_tag, face)
                volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                volume = np.asarray(volume, dtype='uint64')
                self.b[0][v_ids[volume[0]]] += -face_flow

            if face in self.dirichlet_faces:
                volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                volume = np.asarray(volume, dtype='uint64')
                volume_centroid = self.mtu.get_average_position(volume)

                I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
                g_I = self.mb.tag_get_data(self.dirichlet_tag, I)
                g_J = self.mb.tag_get_data(self.dirichlet_tag, J)
                g_K = self.mb.tag_get_data(self.dirichlet_tag, K)
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                face_centroid = self.mtu.get_average_position([face])

                test_vector = face_centroid - volume_centroid
                N_IJK = self.area_vector(JK, JI, test_vector)
                face_area = np.sqrt(np.dot(N_IJK, N_IJK))

                JR = volume_centroid - self.mb.get_coords([J])
                h_R = np.absolute(np.dot(N_IJK, JR) / np.sqrt(np.dot(N_IJK,
                                  N_IJK)))
                tan_JI = np.cross(JI, N_IJK)
                tan_JK = np.cross(N_IJK, JK)

                K_R = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])

                K_R_n = self._flux_term(N_IJK, K_R, N_IJK, face_area)
                K_R_JI = self._flux_term(N_IJK, K_R, tan_JI, face_area)
                K_R_JK = self._flux_term(N_IJK, K_R, tan_JK, face_area)

                D_JI = self._boundary_cross_term(tan_JK, JR, face_area,
                                                 K_R_JK, K_R_n, h_R)
                D_JK = self._boundary_cross_term(tan_JI, JR, face_area,
                                                 K_R_JI, K_R_n, h_R)

                K_n_eff = K_R_n / h_R

                RHS = -(D_JI * (g_J - g_I) + D_JK * (g_J - g_K) + 2 *
                        K_R_n / h_R * g_J)

                volume_id = v_ids[volume[0]]
                self.A[volume_id][volume_id] += -2 * K_n_eff
                self.b[0][volume_id] += RHS

            if face in self.intern_faces:
                face_centroid = self.mtu.get_average_position([face])
                left_volume, right_volume = \
                    self.mtu.get_bridge_adjacencies(face, 2, 3)
                left_vol_cent = self.mtu.get_average_position([left_volume])
                right_vol_cent = self.mtu.get_average_position([right_volume])
                # test_LR = left_vol_cent - right_vol_cent

                I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])

                N_IJK = self.area_vector(JK, JI)
                # N_IJK = np.cross(JK, JI) / 2
                face_area = np.sqrt(np.dot(N_IJK, N_IJK))
                # check_left_or_right = np.dot(test_LR, N_IJK)
                # if check_left_or_right < 0:
                #     right_volume, left_volume = left_volume, right_volume
                #     right_vol_cent, left_vol_cent = left_vol_cent, right_vol_cent

                dist_LR = right_vol_cent - left_vol_cent
                tan_JI = np.cross(JI, N_IJK)
                tan_JK = np.cross(N_IJK, JK)

                K_R = self.mb.tag_get_data(self.perm_tag, right_volume).reshape([3, 3])
                h_R = face_centroid - right_vol_cent
                h_R = np.absolute(np.dot(N_IJK, h_R) / np.sqrt(np.dot(N_IJK,
                                                                      N_IJK)))

                K_R_n = self._flux_term(N_IJK, K_R, N_IJK, face_area)
                K_R_JI = self._flux_term(N_IJK, K_R, tan_JI, face_area)
                K_R_JK = self._flux_term(N_IJK, K_R, tan_JK, face_area)

                K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape([3, 3])
                h_L = face_centroid - left_vol_cent
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

                self.A[gid_right][gid_right] += -2 * K_n_eff
                self.A[gid_right][gid_left] += 2 * K_n_eff

                self.A[gid_left][gid_left] += -2 * K_n_eff
                self.A[gid_left][gid_right] += 2 * K_n_eff

                self._node_treatment(I, nodes_weights, gid_right, gid_left,
                                     v_ids, K_n_eff, D_JI)
                self._node_treatment(K, nodes_weights, gid_right, gid_left,
                                     v_ids, K_n_eff, D_JK)
                self._node_treatment(J, nodes_weights, gid_right, gid_left,
                                     v_ids, K_n_eff, D_JI, cross_2nd=D_JK,
                                     is_J=-1)

        p = np.linalg.solve(self.A, self.b[0])
        # print("PRESSAO: ", p)
        self.mb.tag_set_data(self.pressure_tag, self.volumes, p)
        # self.mb.write_file("pressure_solution.vtk")


class InterpolMethod:

    def __init__(self, mesh_data):
        self.mesh_data = mesh_data
        self.mb = mesh_data.mb
        self.mtu = mesh_data.mtu
        self.dirichlet_tag = mesh_data.dirichlet_tag
        self.neumann_tag = mesh_data.neumann_tag
        self.perm_tag = mesh_data.perm_tag

        self.dirichlet_nodes = set(self.mb.get_entities_by_type_and_tag(
            0, types.MBVERTEX, self.dirichlet_tag, np.array((None,))))

        self.neumann_nodes = set(self.mb.get_entities_by_type_and_tag(
            0, types.MBVERTEX, self.neumann_tag, np.array((None,))))
        self.neumann_nodes = self.neumann_nodes - self.dirichlet_nodes

        boundary_nodes = (self.dirichlet_nodes | self.neumann_nodes)
        self.intern_nodes = set(mesh_data.all_nodes) - boundary_nodes

        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces

        self.all_faces = self.mb.get_entities_by_dimension(0, 2)
        boundary_faces = (self.dirichlet_faces | self.neumann_faces)
        # print('ALL FACES', all_faces, len(all_faces))
        self.intern_faces = set(self.all_faces) - boundary_faces

    def _get_volumes_sharing_face_and_node(self, node, volume):
        faces_in_volume = set(self.mtu.get_bridge_adjacencies(
                                          volume, 3, 2))
        faces_sharing_vertice = set(self.mtu.get_bridge_adjacencies(
                                                node, 0, 2))
        faces_sharing_vertice = faces_in_volume.intersection(
                                            faces_sharing_vertice)
        adj_vols = []
        for face in faces_sharing_vertice:
            volumes_sharing_face = set(self.mtu.get_bridge_adjacencies(
                                                face, 2, 3))
            side_volume = volumes_sharing_face - {volume}
            adj_vols.append(side_volume)
        return adj_vols

    def _get_opposite_node_to_adj_volume(self, volume, adj_volume):
        verts_in_volume = set(self.mtu.get_bridge_adjacencies(
                                            volume, 3, 0))
        verts_in_adj_volume = set(self.mtu.get_bridge_adjacencies(
                                            adj_volume[0], 3, 0))
        opposite_vert = list(verts_in_volume -
                             verts_in_volume.intersection(verts_in_adj_volume))
        return opposite_vert

    def _get_auxiliar_adjacent_volume(self, node, volume, opposite_vert,
                                      adj_volume):
        volumes_around_node = self.mtu.get_bridge_adjacencies(node, 0, 3)
        volumes_around_opposite_vert = set(self.mtu.get_bridge_adjacencies(
                                           opposite_vert[0], 0, 3))
        volumes_around_adj_volume = set(self.mtu.get_bridge_adjacencies(
                                        adj_volume[0], 2, 3))
        auxiliar_adjacent_volume = (volumes_around_opposite_vert.intersection(
                                    volumes_around_node).intersection(
                                    volumes_around_adj_volume) - {volume})
        return auxiliar_adjacent_volume

    def _get_vert_shared_by_volumes(self, node, adj_volume, aux_volume_1,
                                    aux_volume_2):
        verts_in_adj_volume = set(self.mtu.get_bridge_adjacencies(
                                            adj_volume[0], 3, 0))
        verts_in_aux_volume_1 = set(self.mtu.get_bridge_adjacencies(
                                            aux_volume_1[0], 3, 0))
        verts_in_aux_volume_2 = set(self.mtu.get_bridge_adjacencies(
                                            aux_volume_2[0], 3, 0))
        aux_vert = verts_in_adj_volume.intersection(
                   verts_in_aux_volume_1).intersection(
                   verts_in_aux_volume_2) - {node}
        return aux_vert

    def weight(self, volume):
        volume_weight = (A1(volume) * B4(volume) + C2(volume) * D4(volume) +
                         E3(volume) * F4(volume) + G4(volume) * H4(volume) -
                         B2(volume) - F2(volume) - H2(volume))
        return volume_weight

    def by_least_squares(self, node):
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

        G = I_xx * (I_yy*I_zz - I_yz*I_yz) + \
            I_xy * (I_yz*I_xz - I_xy*I_zz) + \
            I_xz * (I_xy*I_yz - I_yy*I_xz)

        psi_x = (R_x * (I_yz*I_yz - I_yy*I_zz) +
                 R_y * (I_xy*I_zz - I_yz*I_xz) +
                 R_z * (I_yy*I_xz - I_xy*I_yz)) / G

        psi_y = (R_x * (I_xy*I_zz - I_yz*I_xz) +
                 R_y * (I_xz*I_xz - I_xx*I_zz) +
                 R_z * (I_xx*I_yz - I_xy*I_xz)) / G

        psi_z = (R_x * (I_yy*I_xz - I_xy*I_yz) +
                 R_y * (I_xx*I_yz - I_xy*I_xz) +
                 R_z * (I_xy*I_xy - I_xx*I_yy)) / G

        num_vols = len(vols_around)
        nodes_weights = {}

        for x_k, y_k, z_k, volume in rel_vol_position:
            numerator = 1.0 + x_k*psi_x + y_k*psi_y + z_k*psi_z
            denominator = num_vols + R_x*psi_x + R_y*psi_y + R_z*psi_z
            nodes_weights[volume] = numerator / denominator

        return nodes_weights

    def by_volumes(self):
        pass

    def by_inverse_distance(self, node):
        coords_node = self.mb.get_coords([node])
        vols_around = self.mtu.get_bridge_adjacencies(node, 0, 3)
        weights = np.array([])
        weight_sum = 0.0
        for a_volume in vols_around:
            vol_cent = self.mesh_data.get_centroid(a_volume)
            inv_dist = 1/self.mesh_data.point_distance(coords_node, vol_cent)
            weights = np.append(weights, inv_dist)
            weight_sum += inv_dist
        weights = weights / weight_sum
        node_weights = {
            vol: weight for vol, weight in zip(vols_around, weights)}
        return node_weights

    def by_lpew2(self, node):
        if node in self.dirichlet_nodes:
            print('a dirichlet node')
        if node in self.neumann_nodes:
            print('a neumann node')
        elif node in self.intern_nodes:
            vols_around = self.mtu.get_bridge_adjacencies(node, 0, 3)
            weights = np.array([])
            weight_sum = 0.0
            for a_vol in vols_around:
                adj_vols = self._get_volumes_sharing_face_and_node(node, a_vol)
                M = a_vol
                W = list(adj_vols.pop())
                R = list(adj_vols.pop())
                L = list(adj_vols.pop())

                i = self._get_opposite_node_to_adj_volume(M, W)
                j = self._get_opposite_node_to_adj_volume(M, R)
                k = self._get_opposite_node_to_adj_volume(M, L)

                N = list(self._get_auxiliar_adjacent_volume(node, M, i, L))
                P = list(self._get_auxiliar_adjacent_volume(node, M, i, R))
                O = list(self._get_auxiliar_adjacent_volume(node, M, k, R))
                S = list(self._get_auxiliar_adjacent_volume(node, M, k, W))
                H = list(self._get_auxiliar_adjacent_volume(node, M, j, L))
                V = list(self._get_auxiliar_adjacent_volume(node, M, j, W))

                w = self._get_vert_shared_by_volumes(node, L, H, N)
                l = self._get_vert_shared_by_volumes(node, R, P, O)
                r = self._get_vert_shared_by_volumes(node, W, S, V)
                print(w, l, r)

                # weights = np.append(weights, weight(a_vol))
