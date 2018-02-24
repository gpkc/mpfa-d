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

        self.intern_nodes = set(mesh_data.all_nodes) - (self.dirichlet_nodes | self.neumann_nodes)

        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces

        self.all_faces = self.mb.get_entities_by_dimension(0, 2)
        # print('ALL FACES', all_faces, len(all_faces))
        self.intern_faces = set(self.all_faces) - (self.dirichlet_faces | self.neumann_faces)

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

    def run(self):

        v_ids = self.set_global_id()

        for a_node in self.intern_nodes | self.neumann_nodes:
            a_node_coords = self.mb.get_coords([a_node])[0]
            self.mb.tag_set_data(self.dirichlet_tag, a_node, 1.0 - a_node_coords)

        volumes = self.mesh_data.all_volumes
        A = np.zeros([len(volumes), len(volumes)])
        b = np.zeros([1, len(volumes)])

        for face in self.all_faces:

            if face in self.neumann_faces:
                face_flow = self.mb.tag_get_data(self.neumann_tag, face)
                volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                volume = np.asarray(volume, dtype='uint64')
                b[0][v_ids[volume[0]]] += -face_flow

            if face in self.dirichlet_faces:
                volume = np.asarray(self.mtu.get_bridge_adjacencies(face, 2, 3), dtype='uint64')
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
                area = np.sqrt(np.dot(N_IJK, N_IJK))

                JR = volume_centroid - self.mb.get_coords([J])
                h_R = np.absolute(np.dot(N_IJK, JR) / np.sqrt(np.dot(N_IJK,
                                  N_IJK)))
                tan_JI = np.cross(JI, N_IJK)
                tan_JK = np.cross(N_IJK, JK)

                K_R = self.mb.tag_get_data(self.perm_tag, volume).reshape([3, 3])
                K_R_n = np.dot(np.dot(np.transpose(N_IJK), K_R), N_IJK) / area

                K_R_JI = np.dot(np.dot(np.transpose(N_IJK), K_R),
                                tan_JI) / area
                K_R_JK = np.dot(np.dot(np.transpose(N_IJK), K_R),
                                tan_JK) / area
                D_JI = ((np.dot(np.cross(JK, N_IJK), JR) / np.dot(N_IJK,
                        N_IJK)) * K_R_n / h_R - K_R_JK / area)
                D_JK = ((np.dot(np.cross(N_IJK, JI), JR) / np.dot(N_IJK,
                        N_IJK)) * K_R_n / h_R - K_R_JI / area)

                K_n_eff = K_R_n / h_R
                RHS = -(D_JI * (g_J - g_I) + D_JK * (g_J - g_K) + 2 *
                        K_R_n / h_R * g_J)

                volume_id = v_ids[volume[0]]
                A[volume_id][volume_id] += -2 * K_n_eff
                b[0][volume_id] += RHS

            if face in self.intern_faces:
                face_centroid = self.mtu.get_average_position([face])
                left_volume, right_volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                left_volume_centroid = self.mtu.get_average_position([left_volume])
                right_volume_centroid = self.mtu.get_average_position([right_volume])
                # test_LR = left_volume_centroid - right_volume_centroid

                I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
                p_I = self.mb.tag_get_data(self.dirichlet_tag, I)
                p_J = self.mb.tag_get_data(self.dirichlet_tag, J)
                p_K = self.mb.tag_get_data(self.dirichlet_tag, K)
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])

                N_IJK = self.area_vector(JK, JI)
                # N_IJK = np.cross(JK, JI) / 2
                area = np.sqrt(np.dot(N_IJK, N_IJK))
                # check_left_or_right = np.dot(test_LR, N_IJK)
                # if check_left_or_right < 0:
                #     right_volume, left_volume = left_volume, right_volume
                #     right_volume_centroid, left_volume_centroid = left_volume_centroid, right_volume_centroid

                LR = right_volume_centroid - left_volume_centroid
                tan_JI = np.cross(JI, N_IJK)
                tan_JK = np.cross(N_IJK, JK)

                K_R = self.mb.tag_get_data(self.perm_tag, right_volume).reshape([3, 3])
                h_R = face_centroid - right_volume_centroid
                h_R = np.absolute(np.dot(N_IJK, h_R) / np.sqrt(np.dot(N_IJK,
                                                                      N_IJK))
                                  )
                K_R_n = np.dot(np.dot(np.transpose(N_IJK), K_R), N_IJK) / area
                K_R_JI = np.dot(np.dot(np.transpose(N_IJK), K_R), tan_JI) / area
                K_R_JK = np.dot(np.dot(np.transpose(N_IJK), K_R), tan_JK) / area

                K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape([3, 3])
                h_L = face_centroid - left_volume_centroid
                h_L = np.absolute(np.dot(N_IJK, h_L) / np.sqrt(np.dot(N_IJK,
                                                                      N_IJK))
                                  )
                K_L_n = np.dot(np.dot(np.transpose(N_IJK), K_L), N_IJK) / area
                K_L_JI = np.dot(np.dot(np.transpose(N_IJK), K_L), tan_JI) / area
                K_L_JK = np.dot(np.dot(np.transpose(N_IJK), K_L), tan_JK) / area
                D_JI = ((np.dot(np.cross(JK, N_IJK), LR) / (np.dot(N_IJK, N_IJK)))
                        - 1 / np.sqrt(np.dot(N_IJK, N_IJK)) * (K_R_JK / K_R_n * h_R
                        + K_L_JK / K_L_n * h_L)
                        )
                D_JK = ((np.dot(np.cross(N_IJK, JI), LR) / (np.dot(N_IJK, N_IJK)))
                        - 1 / np.sqrt(np.dot(N_IJK, N_IJK)) * (K_R_JI / K_R_n * h_R
                        + K_L_JI / K_L_n * h_L)
                        )

                K_n_eff = K_R_n * K_L_n / (K_R_n * h_L + K_L_n * h_R)
                RHS = K_n_eff * (D_JI * (p_I - p_J) + D_JK * (p_K - p_J))

                gid_right = v_ids[right_volume]
                gid_left = v_ids[left_volume]

                A[gid_right][gid_right] += -2 * K_n_eff
                A[gid_right][gid_left] += 2 * K_n_eff

                b[0][gid_right] += RHS

                A[gid_left][gid_left] += -2 * K_n_eff
                A[gid_left][gid_right] += 2 * K_n_eff

                b[0][gid_left] += -RHS

        p = np.linalg.solve(A, b[0])
        print("PRESSAO: ", p)
        self.mb.tag_set_data(self.pressure_tag, volumes, p)
        self.mb.write_file("pressure_solution.vtk")

class InterpolMethod:

    def by_least_squares(self):
        pass

    def by_volumes(self):
        pass

    def by_inverse_distance(self):
        pass

    def by_volumes(self):
        pass

    def by_lpew2(self):
        pass
