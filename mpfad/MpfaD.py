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
        self.source_tag = mesh_data.source_tag

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
        self.intern_nodes = set(self.mesh_data.all_nodes) - boundary_nodes

        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces

        self.all_faces = mesh_data.all_faces
        boundary_faces = (self.dirichlet_faces | self.neumann_faces)
        # print('ALL FACES', all_faces, len(all_faces))
        self.intern_faces = set(self.all_faces) - boundary_faces

        self.volumes = self.mesh_data.all_volumes

        # use pytrilinos to enhance performance
        self.A = lil_matrix((len(self.volumes), len(self.volumes)),
                            dtype=np.float)

        # self.A = np.zeros([len(self.volumes), len(self.volumes)])
        # print('CRIOU MATRIZ A')
        self.b = lil_matrix((len(self.volumes), 1),
                            dtype=np.float)
        # self.b = np.zeros([1, len(self.volumes)])

    # #this must be part of preprocessing
    def _benchmark_1(self, x, y, z):
        K = [1.0, 0.5, 0.0,
             0.5, 1.0, 0.5,
             0.0, 0.5, 10.0]
        y = y + 1/2.
        z = z + 1/3.
        u1 = 1 + np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z)
        return K, x # u1
    #
    # def _benchmark_2(self, x, y, z):
    #     k_xx = y ** 2 + z ** 2 + 1
    #     k_xy = - x * y
    #     k_xz = - x * z
    #     k_yx = - x * y
    #     k_yy = x ** 2 + z ** 2 + 1
    #     k_yz = - y * z
    #     k_zx = - x * z
    #     k_zy = - y * z
    #     k_zz = x ** 2 + y ** 2 + 1
    #
    #     K = [k_xx, k_xy, k_xz,
    #          k_yx, k_yy, k_yz,
    #          k_zx, k_zy, k_zz]
    #     # K = [1 + x, 0.0, 0.0,
    #     #      0.0, 1 + y, 0.0,
    #     #      0.0, 0.0, 1 + z]
    #
    #     u2 = ((x ** 3 * y ** 2 * z) + x * np.sin(2 * pi * x * z)
    #                                     * np.sin(2 * pi * x * y)
    #                                     * np.sin(2 * pi * z))
    #
    #     return K, u2

    def set_global_id(self):
        vol_ids = {}
        range_of_ids = range(len(self.mesh_data.all_volumes))
        for id_, volume in zip(range_of_ids, self.mesh_data.all_volumes):
            vol_ids[volume] = id_
            self.normal_vol_tag = self.mb.tag_get_handle(
                "Normal vol {0}".format(volume), 3,
                types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
            self.LR_tag = self.mb.tag_get_handle(
                "LR {0}".format(volume), 3,
                types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        return vol_ids

    def get_boundary_node_pressure(self, node):
        pressure = self.mesh_data.mb.tag_get_data(self.dirichlet_tag, node)[0]
        return pressure

    def vmv_multiply(self, normal_vector, tensor, CD):
        vmv = (np.dot(np.dot(normal_vector, tensor), CD) /
               np.dot(normal_vector, normal_vector))
        return vmv

    def get_cross_diffusion_term(self, tan, vec, S, h1,
                                 Kn1, Kt1, h2=0, Kt2=0,
                                 Kn2=0, boundary=False):
        mesh_anisotropy_term = (np.dot(tan, vec)/(S ** 2))
        physical_anisotropy_term = -((1 / S) * (h1 * (Kt1 / Kn1)
                                     + h2 * (Kt2 / Kn2)))
        cross_diffusion_term = mesh_anisotropy_term + physical_anisotropy_term
        if boundary:
            dot_term = np.dot(-tan, vec) * Kn1
            cdf_term = h1 * S * Kt1
            b_cross_difusion_term = (dot_term + cdf_term) / (2 * h1 * S)
            return b_cross_difusion_term
        return cross_diffusion_term

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

    def _node_treatment(self, node, id_left, id_right, v_ids,
                        K_eq, D_JK=0, D_JI=0.0):
        RHS = K_eq * (0.5 * D_JK + 0.5 * D_JI)
        if node in self.dirichlet_nodes:
            pressure = self.get_boundary_node_pressure(node)

            # self.b[0][id_left] += RHS * pressure
            # self.b[0][id_right] += -RHS * pressure

            self.b[id_left, 0] += RHS * pressure
            self.b[id_right, 0] += -RHS * pressure


        if node in self.intern_nodes:
            for volume, weight in self.nodes_ws[node].items():
                v_id = v_ids[volume]

                # self.A[id_left][v_id] += -RHS * weight
                # self.A[id_right][v_id] += RHS * weight

                self.A[id_left, v_id] += -RHS * weight
                self.A[id_right, v_id] += RHS * weight

        if node in self.neumann_nodes:
            neu_term = self.nodes_nts[node]

            # self.b[0][id_left] += RHS * neu_term
            # self.b[0][id_right] += -RHS * neu_term

            self.b[id_left, 0] += RHS * neu_term
            self.b[id_right, 0] += -RHS * neu_term

            for volume, weight_N in self.nodes_ws[node].items():
                v_id = v_ids[volume]

                # self.A[id_left][v_id] += -RHS * weight_N
                # self.A[id_right][v_id] += RHS * weight_N

                self.A[id_left, v_id] += -RHS * weight_N
                self.A[id_right, v_id] += RHS * weight_N

    def run_solver(self, interpolation_method):
        node_interpolation = True
        if node_interpolation:
            self.get_nodes_weights(interpolation_method)
        v_ids = self.set_global_id()

        try:
            for volume in self.volumes:
                volume_id = v_ids[volume]
                source_term = self.mb.tag_get_data(self.source_tag, volume)
                # self.b[0][volume_id] += source_term[0][0]
                self.b[volume_id, 0] += source_term[0][0]
        except:
            pass

        for face in self.all_faces:

            if face in self.neumann_faces:
                face_flow = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
                volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                volume = np.asarray(volume, dtype='uint64')
                id_volume = v_ids[volume[0]]
                face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
                node_crds = self.mb.get_coords(face_nodes).reshape([3, 3])
                face_area = geo._area_vector(node_crds, norma=True)

                # self.b[0][id_volume] += - face_flow * face_area
                self.b[id_volume, 0] += - face_flow * face_area

            if face in self.dirichlet_faces:
                I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
                left_volume = np.asarray(self.mtu.get_bridge_adjacencies(
                                         face, 2, 3), dtype='uint64')
                id_volume = v_ids[left_volume[0]]
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                LJ = self.mb.get_coords([J]) - self.mb.get_coords(left_volume)
                N_IJK = np.cross(JI, JK) / 2.
                test = np.dot(LJ, N_IJK)
                if test < 0.:
                    I, K = K, I
                    JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                    JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                    N_IJK = np.cross(JI, JK) / 2.

                tan_JI = np.cross(N_IJK, JI)
                tan_JK = np.cross(N_IJK, JK)
                face_area = np.sqrt(np.dot(N_IJK, N_IJK))

                K_L = self.mb.tag_get_data(self.perm_tag,
                                           left_volume).reshape([3, 3])
                h_L = geo.get_height(N_IJK, LJ)

                g_I = pressure = self.get_boundary_node_pressure(I)
                g_J = pressure = self.get_boundary_node_pressure(J)
                g_K = pressure = self.get_boundary_node_pressure(K)

                K_n_L = self.vmv_multiply(N_IJK, K_L, N_IJK)
                K_L_JI = self.vmv_multiply(N_IJK, K_L, tan_JI)
                K_L_JK = self.vmv_multiply(N_IJK, K_L, tan_JK)

                RHS = ((1/(2 * h_L * face_area)) * ((np.dot(-tan_JK, LJ) *
                       K_n_L + h_L * face_area * K_L_JK) * (g_I - g_J) - 2 *
                       (face_area ** 2) * K_n_L * g_J + (np.dot(-tan_JI, LJ) *
                       K_n_L + h_L * face_area * K_L_JI) * (g_J - g_K)))
                LHS = (1 / h_L)*(face_area * K_n_L)

                # self.A[id_volume][id_volume] += LHS
                # self.b[0][id_volume] += -RHS

                self.A[id_volume, id_volume] += LHS
                self.b[id_volume, 0] += -RHS

            if face in self.intern_faces:
                left_volume, right_volume = \
                    self.mtu.get_bridge_adjacencies(face, 2, 3)
                L = self.mtu.get_average_position([left_volume])
                R = self.mtu.get_average_position([right_volume])
                dist_LR = R - L
                I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])

                N_IJK = np.cross(JI, JK) / 2.
                face_nodes = self.mb.get_coords(self.mtu.get_bridge_adjacencies(face, 0, 0))
                face_nodes = np.reshape(face_nodes, (3, 3))
                test = np.dot(N_IJK, dist_LR)

                if test < 0:
                    left_volume, right_volume = right_volume, left_volume
                    L = self.mtu.get_average_position([left_volume])
                    R = self.mtu.get_average_position([right_volume])
                    dist_LR = R - L

                LJ = L - self.mb.get_coords([J])
                RJ = R - self.mb.get_coords([J])
                face_area = np.sqrt(np.dot(N_IJK, N_IJK))
                tan_JI = np.cross(N_IJK, JI)
                tan_JK = np.cross(N_IJK, JK)

                K_R = self.mb.tag_get_data(self.perm_tag,
                                           right_volume).reshape([3, 3])

                h_R = geo.get_height(N_IJK, RJ)

                K_R_n = self.vmv_multiply(N_IJK, K_R, N_IJK)
                K_R_JI = self.vmv_multiply(N_IJK, K_R, tan_JI)
                K_R_JK = self.vmv_multiply(N_IJK, K_R, tan_JK)

                K_L = self.mb.tag_get_data(self.perm_tag,
                                           left_volume).reshape([3, 3])
                h_L_ = np.absolute(np.dot(N_IJK, LJ) / face_area)
                h_L = geo.get_height(N_IJK, LJ)

                K_L_n = self.vmv_multiply(N_IJK, K_L, N_IJK)
                K_L_JI = self.vmv_multiply(N_IJK, K_L, tan_JI)
                K_L_JK = self.vmv_multiply(N_IJK, K_L, tan_JK)

                D_JI = self.get_cross_diffusion_term(tan_JI, dist_LR, face_area,
                                                     h_L, K_L_n, K_L_JI, h_R,
                                                     K_R_JI, K_R_n)
                D_JK = self.get_cross_diffusion_term(tan_JK, dist_LR, face_area,
                                                     h_L, K_L_n, K_L_JK, h_R,
                                                     K_R_JK, K_R_n)
                K_eq = (K_R_n * K_L_n)/(K_R_n * h_L + K_L_n * h_R) * face_area

                gid_right = v_ids[right_volume]
                gid_left = v_ids[left_volume]

                # self.A[gid_right][gid_right] += K_eq
                # self.A[gid_right][gid_left] += -K_eq
                #
                # self.A[gid_left][gid_left] += K_eq
                # self.A[gid_left][gid_right] += -K_eq

                self.A[gid_right, gid_right] += K_eq
                self.A[gid_right, gid_left] += -K_eq

                self.A[gid_left, gid_left] += K_eq
                self.A[gid_left, gid_right] += -K_eq

                if not node_interpolation:
                    bmk = self._benchmark_1
                    x_I, y_I, z_I = self.mb.get_coords([I])
                    p_I = bmk(x_I, y_I, z_I)[1]
                    x_J, y_J, z_J = self.mb.get_coords([J])
                    p_J = bmk(x_J, y_J, z_J)[1]
                    x_K, y_K, z_K = self.mb.get_coords([K])
                    p_K = bmk(x_K, y_K, z_K)[1]

                    RHS = 0.5 * K_eq * (-D_JK * (p_J - p_I) + D_JI * (p_J - p_K))
                    # self.b[0][gid_left] += RHS
                    # self.b[0][gid_right] += -RHS

                    self.b[gid_left, 0] += RHS
                    self.b[gid_right, 0] += -RHS

                if node_interpolation:
                    self._node_treatment(I, gid_left, gid_right,
                                         v_ids, K_eq, D_JK=D_JK)
                    self._node_treatment(K, gid_left, gid_right,
                                         v_ids, K_eq, D_JI=-D_JI)
                    self._node_treatment(J, gid_left, gid_right,
                                         v_ids, K_eq, D_JI=D_JI,
                                         D_JK=-D_JK)

        self.A = self.A.tocsc()
        self.b = self.b.tocsc()

        p = spsolve(self.A, self.b)
        # p = np.linalg.solve(self.A, self.b[0])

        self.mb.tag_set_data(self.pressure_tag, self.volumes, p)

    def record_data(self, file_name):
        volumes = self.mb.get_entities_by_dimension(0, 3)
        ms = self.mb.create_meshset()
        self.mb.add_entities(ms, volumes)
        self.mb.write_file(file_name, [ms])
