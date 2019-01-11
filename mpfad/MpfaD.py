from pymoab import types
from PyTrilinos import Epetra, AztecOO
import mpfad.helpers.geometric as geo
import numpy as np


class MpfaD3D:

    def __init__(self, mesh_data):

        self.mesh_data = mesh_data
        self.mb = mesh_data.mb
        self.mtu = mesh_data.mtu

        self.comm = Epetra.PyComm()

        self.dirichlet_tag = mesh_data.dirichlet_tag
        self.neumann_tag = mesh_data.neumann_tag
        self.perm_tag = mesh_data.perm_tag
        self.source_tag = mesh_data.source_tag
        self.global_id_tag = mesh_data.global_id_tag
        self.volume_centre_tag = mesh_data.volume_centre_tag
        # self.water_saturation = two_phase.water_saturation

        self.pressure_tag = self.mb.tag_get_handle(
            "Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

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

        std_map = Epetra.Map(len(self.volumes), 0, self.comm)
        self.A_prime = Epetra.CrsMatrix(Epetra.Copy, std_map, 0)
        self.b_prime = Epetra.Vector(std_map)
        self.x_prime = Epetra.Vector(std_map)

    def calculate_relative_perm(self, parameters, water_saturation):
        pass

    def record_data(self, file_name):
        volumes = self.mb.get_entities_by_dimension(0, 3)
        ms = self.mb.create_meshset()
        self.mb.add_entities(ms, volumes)
        self.mb.write_file(file_name, [ms])

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
        if not boundary:
            physical_anisotropy_term = -((1 / S) * (h1 * (Kt1 / Kn1)
                                         + h2 * (Kt2 / Kn2)))
            cross_diffusion_term = (mesh_anisotropy_term +
                                    physical_anisotropy_term)
            return cross_diffusion_term
        if boundary:
            dot_term = np.dot(-tan, vec) * Kn1
            cdf_term = h1 * S * Kt1
            b_cross_difusion_term = (dot_term + cdf_term) / (2 * h1 * S)
            return b_cross_difusion_term

    def get_nodes_weights(self, method):
        self.nodes_ws = {}
        self.nodes_nts = {}
        count = 1
        for node in self.intern_nodes:
            print("{0} / {1}".format(count, len(self.intern_nodes)))
            count += 1
            self.nodes_ws[node] = method(node)
        for node in self.neumann_nodes:
            self.nodes_ws[node] = method(node, neumann=True)
            self.nodes_nts[node] = self.nodes_ws[node].pop(node)

    def _node_treatment(self, node, id_left, id_right, K_eq, D_JK=0, D_JI=0.0):
        RHS = 0.5 * K_eq * (D_JK + D_JI)
        if node in self.dirichlet_nodes:
            pressure = self.get_boundary_node_pressure(node)
            self.b_prime[id_left] += RHS * pressure
            self.b_prime[id_right] += -RHS * pressure

        if node in self.intern_nodes:
            for volume, weight in self.nodes_ws[node].items():
                v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]

                self.A_prime.InsertGlobalValues([id_left, id_right],
                                                [v_id, v_id], [-RHS * weight,
                                                               RHS * weight])

        if node in self.neumann_nodes:
            neu_term = self.nodes_nts[node]

            self.b_prime[id_right] += -RHS * neu_term
            self.b_prime[id_left] += RHS * neu_term

            for volume, weight_N in self.nodes_ws[node].items():
                v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]

                self.A_prime.InsertGlobalValues([id_left, id_right],
                                                [v_id, v_id], [-RHS * weight_N,
                                                               RHS * weight_N])

    def run_solver(self, interpolation_method):

        # TODO: eliminate if cond after debugging
        node_interpolation = True
        if node_interpolation:
            self.get_nodes_weights(interpolation_method)

        try:  # Do this in list comrpehension style
            for volume in self.volumes:
                volume_id = self.mb.tag_get_data(self.global_id_tag,
                                                 volume)[0][0]
                RHS = self.mb.tag_get_data(self.source_tag, volume)[0][0]
                self.b_prime[volume_id] += RHS
        except:
            pass

        for face in self.all_faces:

            if face in self.neumann_faces:
                face_flow = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
                volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
                volume = np.asarray(volume, dtype='uint64')
                id_volume = self.mb.tag_get_data(self.global_id_tag,
                                                 volume)[0][0]
                face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
                node_crds = self.mb.get_coords(face_nodes).reshape([3, 3])
                face_area = geo._area_vector(node_crds, norma=True)
                RHS = face_flow * face_area

                self.b_prime[id_volume] += - RHS

            if face in self.dirichlet_faces:
                I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
                left_volume = np.asarray(self.mtu.get_bridge_adjacencies(
                                         face, 2, 3), dtype='uint64')
                id_volume = self.mb.tag_get_data(self.global_id_tag,
                                                 left_volume)[0][0]
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                LJ = self.mb.get_coords([J]) - \
                    self.mesh_data.mb.tag_get_data(self.volume_centre_tag,
                                                   left_volume)[0]
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

                g_I = self.get_boundary_node_pressure(I)
                g_J = self.get_boundary_node_pressure(J)
                g_K = self.get_boundary_node_pressure(K)

                K_n_L = self.vmv_multiply(N_IJK, K_L, N_IJK)
                K_L_JI = self.vmv_multiply(N_IJK, K_L, tan_JI)
                K_L_JK = self.vmv_multiply(N_IJK, K_L, tan_JK)

                K_n_L = self.vmv_multiply(N_IJK, K_L, N_IJK)
                K_L_JI = self.vmv_multiply(N_IJK, K_L, tan_JI)
                K_L_JK = self.vmv_multiply(N_IJK, K_L, tan_JK)

                D_JK = self.get_cross_diffusion_term(tan_JK, LJ, face_area,
                                                     h_L, K_n_L, K_L_JK,
                                                     boundary=True)
                D_JI = self.get_cross_diffusion_term(tan_JI, LJ, face_area,
                                                     h_L, K_n_L, K_L_JI,
                                                     boundary=True)
                K_eq = (1 / h_L)*(face_area * K_n_L)

                RHS = (D_JK * (g_I - g_J) - K_eq * g_J + D_JI * (g_J - g_K))
                LHS = K_eq

                self.A_prime.InsertGlobalValues(id_volume, id_volume, LHS)
                self.b_prime[id_volume] += -RHS

            if face in self.intern_faces:
                left_volume, right_volume = \
                    self.mtu.get_bridge_adjacencies(face, 2, 3)
                L = self.mesh_data.mb.tag_get_data(self.volume_centre_tag,
                                                   left_volume)[0]
                R = self.mesh_data.mb.tag_get_data(self.volume_centre_tag,
                                                   right_volume)[0]
                dist_LR = R - L
                I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])

                N_IJK = np.cross(JI, JK) / 2.
                face_nodes = self.mb.get_coords(
                    self.mtu.get_bridge_adjacencies(face, 0, 0))
                face_nodes = np.reshape(face_nodes, (3, 3))
                test = np.dot(N_IJK, dist_LR)

                if test < 0:
                    left_volume, right_volume = right_volume, left_volume
                    L = self.mesh_data.mb.tag_get_data(self.volume_centre_tag,
                                                       left_volume)[0]
                    R = self.mesh_data.mb.tag_get_data(self.volume_centre_tag,
                                                       right_volume)[0]
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
                h_L = geo.get_height(N_IJK, LJ)

                K_L_n = self.vmv_multiply(N_IJK, K_L, N_IJK)
                K_L_JI = self.vmv_multiply(N_IJK, K_L, tan_JI)
                K_L_JK = self.vmv_multiply(N_IJK, K_L, tan_JK)

                D_JI = self.get_cross_diffusion_term(tan_JI, dist_LR,
                                                     face_area, h_L, K_L_n,
                                                     K_L_JI, h_R,
                                                     K_R_JI, K_R_n)
                D_JK = self.get_cross_diffusion_term(tan_JK, dist_LR,
                                                     face_area, h_L, K_L_n,
                                                     K_L_JK, h_R,
                                                     K_R_JK, K_R_n)
                K_eq = (K_R_n * K_L_n)/(K_R_n * h_L + K_L_n * h_R) * face_area

                id_right = self.mb.tag_get_data(self.global_id_tag,
                                                right_volume)[0][0]
                id_left = self.mb.tag_get_data(self.global_id_tag, left_volume)
                col_ids = [id_right, id_right, id_left, id_left]
                row_ids = [id_right, id_left, id_left, id_right]
                values = [K_eq, -K_eq, K_eq, -K_eq]

                self.A_prime.InsertGlobalValues(col_ids, row_ids, values)

                # TODO: eliminate if block after debugging
                # if not node_interpolation:
                #     bmk = self._benchmark_1
                #     x_I, y_I, z_I = self.mb.get_coords([I])
                #     p_I = bmk(x_I, y_I, z_I)[1]
                #     x_J, y_J, z_J = self.mb.get_coords([J])
                #     p_J = bmk(x_J, y_J, z_J)[1]
                #     x_K, y_K, z_K = self.mb.get_coords([K])
                #     p_K = bmk(x_K, y_K, z_K)[1]
                #     RHS = 0.5 * K_eq * (-D_JK * (p_J - p_I) +
                #                         D_JI * (p_J - p_K))
                #
                #     self.b_prime[id_left] += RHS
                #     self.b_prime[id_right] += -RHS

                self._node_treatment(I, id_left, id_right, K_eq, D_JK=D_JK)
                self._node_treatment(J, id_left, id_right, K_eq,
                                     D_JI=D_JI, D_JK=-D_JK)
                self._node_treatment(K, id_left, id_right, K_eq, D_JI=-D_JI)

        self.A_prime.FillComplete()
        linearProblem = Epetra.LinearProblem(self.A_prime,
                                             self.x_prime,
                                             self.b_prime)
        solver = AztecOO.AztecOO(linearProblem)
        solver.Iterate(1000, 1e-9)
        self.mb.tag_set_data(self.pressure_tag, self.volumes, self.x_prime)
        self.record_data('zones_validation.vtk')
