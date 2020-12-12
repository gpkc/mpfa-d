"""This is the begin."""
from pymoab import types
from PyTrilinos import Epetra, AztecOO, Amesos
import solvers.helpers.geometric as geo
import numpy as np
import time


# from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve
# from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve


class MpfaD3D:
    """Implement the MPFAD method."""

    def __init__(self, mesh_data, x=None, mobility=None):
        """Init class."""
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
        self.pressure_tag = mesh_data.pressure_tag

        self.flux_info_tag = self.mb.tag_get_handle(
            "flux info", 7, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        self.normal_tag = self.mb.tag_get_handle(
            "Normal", 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
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
        self.intern_nodes = set(self.mesh_data.all_nodes) - boundary_nodes

        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces
        self.intern_faces = mesh_data.intern_faces()
        # self.intern_faces = set(mesh_data.all_faces).difference(
        #     self.dirichlet_faces | self.neumann_faces
        # )
        self.volumes = self.mesh_data.all_volumes

        std_map = Epetra.Map(len(self.volumes), 0, self.comm)
        self.T = Epetra.CrsMatrix(Epetra.Copy, std_map, 0)
        self.Q = Epetra.Vector(std_map)
        if x is None:
            self.x = Epetra.Vector(std_map)
        else:
            self.x = x
        # self.T = lil_matrix((len(self.volumes), len(self.volumes)),
        #                     dtype=np.float)
        # self.Q = lil_matrix((len(self.volumes), 1), dtype=np.float)

    def record_data(self, file_name):
        """Record data to file."""
        volumes = self.mb.get_entities_by_dimension(0, 3)
        # faces = self.mb.get_entities_by_dimension(0, 2)
        ms = self.mb.create_meshset()
        self.mb.add_entities(ms, volumes)
        # self.mb.add_entities(ms, faces)
        self.mb.write_file(file_name, [ms])

    def get_boundary_node_pressure(self, node):
        """Return pressure at the boundary nodes of the mesh."""
        pressure = self.mesh_data.mb.tag_get_data(self.dirichlet_tag, node)[0]
        return pressure

    def vmv_multiply(self, normal_vector, tensor, CD):
        """Return a vector-matrix-vector multiplication."""
        vmv = np.dot(np.dot(normal_vector, tensor), CD) / np.dot(
            normal_vector, normal_vector
        )
        return vmv

    def get_cross_diffusion_term(
        self, tan, vec, S, h1, Kn1, Kt1, h2=0, Kt2=0, Kn2=0, boundary=False
    ):
        """Return a cross diffusion multiplication term."""
        if not boundary:
            mesh_anisotropy_term = np.dot(tan, vec) / (S ** 2)
            physical_anisotropy_term = -(
                (1 / S) * (h1 * (Kt1 / Kn1) + h2 * (Kt2 / Kn2))
            )
            cross_diffusion_term = (
                mesh_anisotropy_term + physical_anisotropy_term
            )
            return cross_diffusion_term
        if boundary:
            dot_term = np.dot(-tan, vec) * Kn1
            cdf_term = h1 * S * Kt1
            b_cross_difusion_term = (dot_term + cdf_term) / (2 * h1 * S)
            return b_cross_difusion_term

    # @celery.task
    def get_nodes_weights(self, method):
        """Return the node weights."""
        self.nodes_ws = {}
        self.nodes_nts = {}
        # This is the limiting part of the interpoation method. The Dict
        for node in self.intern_nodes:
            self.nodes_ws[node] = method(node)

        for node in self.neumann_nodes:
            self.nodes_ws[node] = method(node, neumann=True)
            self.nodes_nts[node] = self.nodes_ws[node].pop(node)

    def _node_treatment(self, node, id_left, id_right, K_eq, D_JK=0, D_JI=0.0):
        """Add flux term from nodes RHS."""
        RHS = 0.5 * K_eq * (D_JK + D_JI)
        if node in self.dirichlet_nodes:
            pressure = self.get_boundary_node_pressure(node)
            self.Q[id_left] += RHS * pressure
            self.Q[id_right] += -RHS * pressure
            # self.Q[id_left[0], 0] += RHS * pressure
            # self.Q[id_right[0], 0] += - RHS * pressure

        if node in self.intern_nodes:
            for volume, weight in self.nodes_ws[node].items():
                self.ids.append([id_left, id_right])
                v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
                self.v_ids.append([v_id, v_id])
                self.ivalues.append([-RHS * weight, RHS * weight])

        if node in self.neumann_nodes:
            neu_term = self.nodes_nts[node]
            self.Q[id_right] += -RHS * neu_term
            self.Q[id_left] += RHS * neu_term
            # self.Q[id_right, 0] += - RHS * neu_term
            # self.Q[id_left, 0] += RHS * neu_term

            for volume, weight_N in self.nodes_ws[node].items():
                self.ids.append([id_left, id_right])
                v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
                self.v_ids.append([v_id, v_id])
                self.ivalues.append([-RHS * weight_N, RHS * weight_N])

    def run_solver(self, interpolation_method):
        """Run solver."""
        self.interpolation_method = interpolation_method
        t0 = time.time()
        n_vertex = len(set(self.mesh_data.all_nodes) - self.dirichlet_nodes)
        print("interpolation runing...")
        self.get_nodes_weights(interpolation_method)
        print(
            "done interpolation...",
            "took {0} seconds to interpolate over {1} verts".format(
                time.time() - t0, n_vertex
            ),
        )
        print("filling the transmissibility matrix...")
        begin = time.time()

        try:
            for volume in self.volumes:
                volume_id = self.mb.tag_get_data(self.global_id_tag, volume)[
                    0
                ][0]
                RHS = self.mb.tag_get_data(self.source_tag, volume)[0][0]
                self.Q[volume_id] += RHS
                # self.Q[volume_id, 0] += RHS
        except Exception:
            pass

        for face in self.neumann_faces:
            face_flow = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
            volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
            volume = np.asarray(volume, dtype="uint64")
            id_volume = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
            face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
            node_crds = self.mb.get_coords(face_nodes).reshape([3, 3])
            face_area = geo._area_vector(node_crds, norma=True)
            RHS = face_flow * face_area
            self.Q[id_volume] += -RHS
            # self.Q[id_volume, 0] += - RHS

        id_volumes = []
        all_LHS = []
        for face in self.dirichlet_faces:
            # '2' argument was initially '0' but it's incorrect
            I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)

            left_volume = np.asarray(
                self.mtu.get_bridge_adjacencies(face, 2, 3), dtype="uint64"
            )
            id_volume = self.mb.tag_get_data(self.global_id_tag, left_volume)[
                0
            ][0]
            id_volumes.append(id_volume)

            JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
            JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
            LJ = (
                self.mb.get_coords([J])
                - self.mesh_data.mb.tag_get_data(
                    self.volume_centre_tag, left_volume
                )[0]
            )
            N_IJK = np.cross(JI, JK) / 2.0
            _test = np.dot(LJ, N_IJK)
            if _test < 0.0:
                I, K = K, I
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                N_IJK = np.cross(JI, JK) / 2.0
            tan_JI = np.cross(N_IJK, JI)
            tan_JK = np.cross(N_IJK, JK)
            self.mb.tag_set_data(self.normal_tag, face, N_IJK)

            face_area = np.sqrt(np.dot(N_IJK, N_IJK))
            h_L = geo.get_height(N_IJK, LJ)

            g_I = self.get_boundary_node_pressure(I)
            g_J = self.get_boundary_node_pressure(J)
            g_K = self.get_boundary_node_pressure(K)

            K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
                [3, 3]
            )
            K_n_L = self.vmv_multiply(N_IJK, K_L, N_IJK)
            K_L_JI = self.vmv_multiply(N_IJK, K_L, tan_JI)
            K_L_JK = self.vmv_multiply(N_IJK, K_L, tan_JK)

            D_JK = self.get_cross_diffusion_term(
                tan_JK, LJ, face_area, h_L, K_n_L, K_L_JK, boundary=True
            )
            D_JI = self.get_cross_diffusion_term(
                tan_JI, LJ, face_area, h_L, K_n_L, K_L_JI, boundary=True
            )
            K_eq = (1 / h_L) * (face_area * K_n_L)

            RHS = D_JK * (g_I - g_J) - K_eq * g_J + D_JI * (g_J - g_K)
            LHS = K_eq
            all_LHS.append(LHS)

            self.Q[id_volume] += -RHS
            # self.Q[id_volume, 0] += - RHS
            # self.mb.tag_set_data(self.flux_info_tag, face,
            #                      [D_JK, D_JI, K_eq, I, J, K, face_area])

        all_cols = []
        all_rows = []
        all_values = []
        self.ids = []
        self.v_ids = []
        self.ivalues = []
        for face in self.intern_faces:
            left_volume, right_volume = self.mtu.get_bridge_adjacencies(
                face, 2, 3
            )
            L = self.mesh_data.mb.tag_get_data(
                self.volume_centre_tag, left_volume
            )[0]
            R = self.mesh_data.mb.tag_get_data(
                self.volume_centre_tag, right_volume
            )[0]
            dist_LR = R - L
            I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
            JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
            JK = self.mb.get_coords([K]) - self.mb.get_coords([J])

            N_IJK = np.cross(JI, JK) / 2.0
            test = np.dot(N_IJK, dist_LR)

            if test < 0:
                left_volume, right_volume = right_volume, left_volume
                L = self.mesh_data.mb.tag_get_data(
                    self.volume_centre_tag, left_volume
                )[0]
                R = self.mesh_data.mb.tag_get_data(
                    self.volume_centre_tag, right_volume
                )[0]
                dist_LR = R - L

            face_area = np.sqrt(np.dot(N_IJK, N_IJK))
            tan_JI = np.cross(N_IJK, JI)
            tan_JK = np.cross(N_IJK, JK)

            K_R = self.mb.tag_get_data(self.perm_tag, right_volume).reshape(
                [3, 3]
            )
            RJ = R - self.mb.get_coords([J])
            h_R = geo.get_height(N_IJK, RJ)

            K_R_n = self.vmv_multiply(N_IJK, K_R, N_IJK)
            K_R_JI = self.vmv_multiply(N_IJK, K_R, tan_JI)
            K_R_JK = self.vmv_multiply(N_IJK, K_R, tan_JK)

            K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
                [3, 3]
            )

            LJ = L - self.mb.get_coords([J])
            h_L = geo.get_height(N_IJK, LJ)

            K_L_n = self.vmv_multiply(N_IJK, K_L, N_IJK)
            K_L_JI = self.vmv_multiply(N_IJK, K_L, tan_JI)
            K_L_JK = self.vmv_multiply(N_IJK, K_L, tan_JK)

            D_JI = self.get_cross_diffusion_term(
                tan_JI,
                dist_LR,
                face_area,
                h_L,
                K_L_n,
                K_L_JI,
                h_R,
                K_R_JI,
                K_R_n,
            )
            D_JK = self.get_cross_diffusion_term(
                tan_JK,
                dist_LR,
                face_area,
                h_L,
                K_L_n,
                K_L_JK,
                h_R,
                K_R_JK,
                K_R_n,
            )

            K_eq = (K_R_n * K_L_n) / (K_R_n * h_L + K_L_n * h_R) * face_area

            id_right = self.mb.tag_get_data(self.global_id_tag, right_volume)
            id_left = self.mb.tag_get_data(self.global_id_tag, left_volume)

            col_ids = [id_right, id_right, id_left, id_left]
            row_ids = [id_right, id_left, id_left, id_right]
            values = [K_eq, -K_eq, K_eq, -K_eq]
            all_cols.append(col_ids)
            all_rows.append(row_ids)
            all_values.append(values)
            # wait for interpolation to be done
            self._node_treatment(I, id_left, id_right, K_eq, D_JK=D_JK)
            self._node_treatment(
                J, id_left, id_right, K_eq, D_JI=D_JI, D_JK=-D_JK
            )
            self._node_treatment(K, id_left, id_right, K_eq, D_JI=-D_JI)
            # self.mb.tag_set_data(self.flux_info_tag, face,
            #                      [D_JK, D_JI, K_eq, I, J, K, face_area])

        self.T.InsertGlobalValues(self.ids, self.v_ids, self.ivalues)
        # self.T[
        #     np.asarray(self.ids)[:, :, 0, 0], np.asarray(self.v_ids)
        # ] = np.asarray(self.ivalues)
        self.T.InsertGlobalValues(id_volumes, id_volumes, all_LHS)
        # self.T[
        #     np.asarray(id_volumes), np.asarray(id_volumes)
        # ] = np.asarray(all_LHS)
        self.T.InsertGlobalValues(all_cols, all_rows, all_values)
        # self.T[
        #     np.asarray(all_cols)[:, 0, 0, 0],
        #     np.asarray(all_rows)[:, 0, 0, 0]
        # ] = np.asarray(all_values)[:, 0]
        self.T.FillComplete()
        mat_fill_time = time.time() - begin
        print("matrix fill took {0} seconds...".format(mat_fill_time))
        mesh_size = len(self.volumes)
        print("running solver...")
        USE_DIRECT_SOLVER = False
        linearProblem = Epetra.LinearProblem(self.T, self.x, self.Q)
        if USE_DIRECT_SOLVER:
            solver = Amesos.Lapack(linearProblem)
            print("1) Performing symbolic factorizations...")
            solver.SymbolicFactorization()
            print("2) Performing numeric factorizations...")
            solver.NumericFactorization()
            print("3) Solving the linear system...")
            solver.Solve()
            t = time.time() - t0
            print(
                "Solver took {0} seconds to run over {1} volumes".format(
                    t, mesh_size
                )
            )
        else:
            solver = AztecOO.AztecOO(linearProblem)
            solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres)
            solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_none)
            # solver.SetAztecOption(AztecOO.AZ_precond, AztecOO.AZ_Jacobi)
            # solver.SetAztecOption(AztecOO.AZ_kspace, 1251)
            # solver.SetAztecOption(AztecOO.AZ_orthog, AztecOO.AZ_modified)
            # solver.SetAztecOption(AztecOO.AZ_conv, AztecOO.AZ_Anorm)
            solver.Iterate(8000, 1e-10)
            t = time.time() - t0
            its = solver.GetAztecStatus()[0]
            solver_time = solver.GetAztecStatus()[6]
            print(
                "Solver took {0} seconds to run over {1} volumes".format(
                    t, mesh_size
                )
            )
            print(
                "Solver converged at %.dth iteration in %3f seconds."
                % (int(its), solver_time)
            )
        self.mb.tag_set_data(self.pressure_tag, self.volumes, self.x)

        coords_x = [
            coords[0] for coords in self.mb.get_coords(
                self.volumes
            ).reshape([len(self.volumes), 3])
        ]
        with open('results.csv', 'w') as results:
            for coord, x in zip(coords_x, self.x):
                results.write(f"{str(coord)};{str(x)}\n")


# """This is the begin."""
# from pymoab import types
# from PyTrilinos import Epetra, AztecOO, Amesos
# import solvers.helpers.geometric as geo
# import numpy as np
# import time
#
# # from solvers.helpers import matrix_evaluation as eval
#
# # from scipy.sparse import lil_matrix
# # from scipy.sparse.linalg import spsolve
# # from scipy.sparse import lil_matrix
# # from scipy.sparse.linalg import spsolve
#
#
# class MpfaD3D:
#     """Implement the MPFAD method."""
#
#     def __init__(self, mesh_data, x=None, mobility=None):
#         """Init class."""
#         self.mesh_data = mesh_data
#         self.mb = mesh_data.mb
#         self.mtu = mesh_data.mtu
#
#         self.comm = Epetra.PyComm()
#
#         self.dirichlet_tag = mesh_data.dirichlet_tag
#         self.neumann_tag = mesh_data.neumann_tag
#         self.perm_tag = mesh_data.perm_tag
#         self.source_tag = mesh_data.source_tag
#         self.global_id_tag = mesh_data.global_id_tag
#         self.volume_centre_tag = mesh_data.volume_centre_tag
#         self.pressure_tag = mesh_data.pressure_tag
#         self.velocity_tag = mesh_data.velocity_tag
#         self.node_pressure_tag = mesh_data.node_pressure_tag
#         self.face_mobility_tag = mesh_data.face_mobility_tag
#         self.left_volume_tag = mesh_data.left_volume_tag
#
#         self.flux_info_tag = self.mb.tag_get_handle(
#             "flux info", 7, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
#         )
#
#         self.normal_tag = self.mb.tag_get_handle(
#             "Normal", 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
#         )
#
#         self.dirichlet_nodes = set(
#             self.mb.get_entities_by_type_and_tag(
#                 0, types.MBVERTEX, self.dirichlet_tag, np.array((None,))
#             )
#         )
#
#         self.neumann_nodes = set(
#             self.mb.get_entities_by_type_and_tag(
#                 0, types.MBVERTEX, self.neumann_tag, np.array((None,))
#             )
#         )
#         self.neumann_nodes = self.neumann_nodes - self.dirichlet_nodes
#
#         boundary_nodes = self.dirichlet_nodes | self.neumann_nodes
#         self.intern_nodes = set(self.mesh_data.all_nodes) - boundary_nodes
#
#         self.dirichlet_faces = mesh_data.dirichlet_faces
#         self.neumann_faces = mesh_data.neumann_faces
#         self.intern_faces = mesh_data.intern_faces()
#         self.volumes = self.mesh_data.all_volumes
#
#         std_map = Epetra.Map(len(self.volumes), 0, self.comm)
#         self.T = Epetra.CrsMatrix(Epetra.Copy, std_map, 0)
#         self.Q = Epetra.Vector(std_map)
#         if x is None:
#             self.x = Epetra.Vector(std_map)
#         else:
#             self.x = x
#         # self.T = lil_matrix((len(self.volumes), len(self.volumes)),
#         #                     dtype=np.float)
#         # self.Q = lil_matrix((len(self.volumes), 1), dtype=np.float)
#
#     def get_grad(self, a_volume):
#         vol_faces = self.mtu.get_bridge_adjacencies(a_volume, 2, 2)
#         vol_nodes = self.mtu.get_bridge_adjacencies(a_volume, 0, 0)
#         vol_crds = self.mb.get_coords(vol_nodes)
#         vol_crds = np.reshape(vol_crds, ([4, 3]))
#         vol_volume = self.mesh_data.get_tetra_volume(vol_crds)
#         I, J, K = self.mtu.get_bridge_adjacencies(vol_faces[0], 2, 0)
#         L = list(
#             set(vol_nodes).difference(
#                 set(
#                     self.mtu.get_bridge_adjacencies(
#                         vol_faces[0], 2, 0
#                     )
#                 )
#             )
#         )
#         JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#         JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
#         LJ = self.mb.get_coords([J]) - self.mb.get_coords(L)
#         N_IJK = np.cross(JI, JK) / 2.0
#
#         test = np.dot(LJ, N_IJK)
#         if test < 0.0:
#             I, K = K, I
#             JI = self.mb.get_coords([I]) - self.mb.get_coords(
#                 [J]
#             )
#             JK = self.mb.get_coords([K]) - self.mb.get_coords(
#                 [J]
#             )
#             N_IJK = np.cross(JI, JK) / 2.0
#
#         tan_JI = np.cross(N_IJK, JI)
#         tan_JK = np.cross(N_IJK, JK)
#         face_area = np.sqrt(np.dot(N_IJK, N_IJK))
#
#         h_L = geo.get_height(N_IJK, LJ)
#
#         p_I = self.mb.tag_get_data(self.node_pressure_tag, I)
#         p_J = self.mb.tag_get_data(self.node_pressure_tag, J)
#         p_K = self.mb.tag_get_data(self.node_pressure_tag, K)
#         p_L = self.mb.tag_get_data(self.node_pressure_tag, L)
#         grad_normal = -2 * (p_J - p_L) * N_IJK
#         grad_cross_I = (p_J - p_I) * (
#             (np.dot(tan_JK, LJ) / face_area ** 2) * N_IJK
#             - (h_L / (face_area)) * tan_JK
#         )
#         grad_cross_K = (p_K - p_J) * (
#             (np.dot(tan_JI, LJ) / face_area ** 2) * N_IJK
#             - (h_L / (face_area)) * tan_JI
#         )
#
#         grad_p = -(1 / (6 * vol_volume)) * (
#             grad_normal + grad_cross_I + grad_cross_K
#         )
#         return grad_p
#
#     def record_data(self, file_name):
#         """Record data to file."""
#         volumes = self.mb.get_entities_by_dimension(0, 3)
#         # faces = self.mb.get_entities_by_dimension(0, 2)
#         ms = self.mb.create_meshset()
#         self.mb.add_entities(ms, volumes)
#         # self.mb.add_entities(ms, faces)
#         self.mb.write_file(file_name, [ms])
#
#     def tag_verts_pressure(self):
#         p_verts = []
#         for node in self.mesh_data.all_nodes:
#             try:
#                 p_vert = self.mb.tag_get_data(self.dirichlet_tag, node)
#                 p_verts.append(p_vert[0])
#             except Exception:
#                 p_vert = 0.0
#                 p_tag = self.pressure_tag
#                 nd_weights = self.nodes_ws[node]
#                 for volume, wt in nd_weights.items():
#                     p_vol = self.mb.tag_get_data(p_tag, volume)
#                     p_vert += p_vol * wt
#                 p_verts.append(p_vert)
#         self.mb.tag_set_data(
#             self.node_pressure_tag, self.mesh_data.all_nodes, p_verts
#         )
#
#     def tag_velocity(self):
#         self.tag_verts_pressure()
#         velocities = []
#         for face in self.neumann_faces:
#             face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
#                 0
#             ][0]
#             face_flow = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
#             volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
#             volume = np.asarray(volume, dtype="uint64")
#             face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
#             node_crds = self.mb.get_coords(face_nodes).reshape([3, 3])
#             face_area = geo._area_vector(node_crds, norma=True)
#             velocity = face_mobility * face_flow * face_area
#             velocities.append(velocity)
#         self.mb.tag_set_data(self.velocity_tag, self.neumann_faces, velocities)
#
#         velocities = []
#         dirichlet_faces = list(self.dirichlet_faces)
#         for face in dirichlet_faces:
#             face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
#                 0
#             ][0]
#             # '2' argument was initially '0' but it's incorrect
#             I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)
#
#             left_volume = np.asarray(
#                 self.mtu.get_bridge_adjacencies(face, 2, 3), dtype="uint64"
#             )
#
#             JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#             JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
#             LJ = (
#                 self.mb.get_coords([J])
#                 - self.mesh_data.mb.tag_get_data(
#                     self.volume_centre_tag, left_volume
#                 )[0]
#             )
#             N_IJK = np.cross(JI, JK) / 2.0
#             _test = np.dot(LJ, N_IJK)
#             if _test < 0.0:
#                 I, K = K, I
#                 JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#                 JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
#                 N_IJK = np.cross(JI, JK) / 2.0
#             tan_JI = np.cross(N_IJK, JI)
#             tan_JK = np.cross(N_IJK, JK)
#             self.mb.tag_set_data(self.normal_tag, face, N_IJK)
#
#             face_area = np.sqrt(np.dot(N_IJK, N_IJK))
#             h_L = geo.get_height(N_IJK, LJ)
#
#             g_I = self.mb.tag_get_data(self.node_pressure_tag, I)
#             g_J = self.mb.tag_get_data(self.node_pressure_tag, J)
#             g_K = self.mb.tag_get_data(self.node_pressure_tag, K)
#
#             K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
#                 [3, 3]
#             )
#             K_n_L = self.vmv_multiply(N_IJK, face_mobility * K_L, N_IJK)
#             K_L_JI = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JI)
#             K_L_JK = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JK)
#
#             D_JK = self.get_cross_diffusion_term(
#                 tan_JK, LJ, face_area, h_L, K_n_L, K_L_JK, boundary=True
#             )
#             D_JI = self.get_cross_diffusion_term(
#                 tan_JI, LJ, face_area, h_L, K_n_L, K_L_JI, boundary=True
#             )
#             K_eq = (1 / h_L) * (face_area * K_n_L)
#             p_vol = self.mb.tag_get_data(self.pressure_tag, left_volume)
#             velocity = (
#                 D_JK * (g_I - g_J) - K_eq * (p_vol - g_J) + D_JI * (g_J - g_K)
#             )
#             velocities.append(velocity)
#         vels = np.asarray(velocities).flatten()
#         self.mb.tag_set_data(self.velocity_tag, dirichlet_faces, vels)
#         velocities = []
#         left_vols = []
#         intern_faces = list(self.intern_faces)
#         for face in intern_faces:
#             face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
#                 0
#             ][0]
#             left_volume, right_volume = self.mtu.get_bridge_adjacencies(
#                 face, 2, 3
#             )
#             L = self.mesh_data.mb.tag_get_data(
#                 self.volume_centre_tag, left_volume
#             )[0]
#             R = self.mesh_data.mb.tag_get_data(
#                 self.volume_centre_tag, right_volume
#             )[0]
#             dist_LR = R - L
#             I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
#             JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#             JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
#
#             N_IJK = np.cross(JI, JK) / 2.0
#             test = np.dot(N_IJK, dist_LR)
#
#             if test < 0:
#                 left_volume, right_volume = right_volume, left_volume
#                 L = self.mesh_data.mb.tag_get_data(
#                     self.volume_centre_tag, left_volume
#                 )[0]
#                 R = self.mesh_data.mb.tag_get_data(
#                     self.volume_centre_tag, right_volume
#                 )[0]
#                 dist_LR = R - L
#
#             face_area = np.sqrt(np.dot(N_IJK, N_IJK))
#             tan_JI = np.cross(N_IJK, JI)
#             tan_JK = np.cross(N_IJK, JK)
#
#             K_R = self.mb.tag_get_data(self.perm_tag, right_volume).reshape(
#                 [3, 3]
#             )
#             RJ = R - self.mb.get_coords([J])
#             h_R = geo.get_height(N_IJK, RJ)
#
#             K_R_n = self.vmv_multiply(N_IJK, face_mobility * K_R, N_IJK)
#             K_R_JI = self.vmv_multiply(N_IJK, face_mobility * K_R, tan_JI)
#             K_R_JK = self.vmv_multiply(N_IJK, face_mobility * K_R, tan_JK)
#
#             K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
#                 [3, 3]
#             )
#
#             LJ = L - self.mb.get_coords([J])
#             h_L = geo.get_height(N_IJK, LJ)
#
#             K_L_n = self.vmv_multiply(N_IJK, face_mobility * K_L, N_IJK)
#             K_L_JI = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JI)
#             K_L_JK = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JK)
#
#             D_JI = self.get_cross_diffusion_term(
#                 tan_JI,
#                 dist_LR,
#                 face_area,
#                 h_L,
#                 K_L_n,
#                 K_L_JI,
#                 h_R,
#                 K_R_JI,
#                 K_R_n,
#             )
#             D_JK = self.get_cross_diffusion_term(
#                 tan_JK,
#                 dist_LR,
#                 face_area,
#                 h_L,
#                 K_L_n,
#                 K_L_JK,
#                 h_R,
#                 K_R_JK,
#                 K_R_n,
#             )
#
#             K_eq = (K_R_n * K_L_n) / (K_R_n * h_L + K_L_n * h_R) * face_area
#             p_r = self.mb.tag_get_data(self.pressure_tag, right_volume)
#             p_l = self.mb.tag_get_data(self.pressure_tag, left_volume)
#             p_I = self.mb.tag_get_data(self.node_pressure_tag, I)
#             p_J = self.mb.tag_get_data(self.node_pressure_tag, J)
#             p_K = self.mb.tag_get_data(self.node_pressure_tag, K)
#             velocity = K_eq * (
#                 p_r - p_l - D_JI * (p_I - p_J) - D_JK * (p_K - p_J)
#             )
#             velocities.append(velocity)
#             left_vols.append(left_volume)
#         velocities = np.asarray(velocities).flatten()
#         self.mb.tag_set_data(self.velocity_tag, intern_faces, velocities)
#         self.mb.tag_set_data(
#             self.left_volume_tag, left_vols, np.repeat(1, len(left_vols))
#         )
#
#     def get_mobility(self):
#         faces = self.mb.get_entities_by_dimension(0, 2)
#         try:
#             self.mb.tag_get_data(self.face_mobility_tag, faces)
#         except RuntimeError:
#             mobility_init = np.repeat(1.0, len(faces))
#             self.mb.tag_set_data(self.face_mobility_tag, faces, mobility_init)
#
#     def get_boundary_node_pressure(self, node):
#         """Return pressure at the boundary nodes of the mesh."""
#         pressure = self.mesh_data.mb.tag_get_data(self.dirichlet_tag, node)[0]
#         return pressure
#
#     def vmv_multiply(self, normal_vector, tensor, CD):
#         """Return a vector-matrix-vector multiplication."""
#         vmv = np.dot(np.dot(normal_vector, tensor), CD) / np.dot(
#             normal_vector, normal_vector
#         )
#         return vmv
#
#     def get_cross_diffusion_term(
#         self, tan, vec, S, h1, Kn1, Kt1, h2=0, Kt2=0, Kn2=0, boundary=False
#     ):
#         """Return a cross diffusion multiplication term."""
#         if not boundary:
#             mesh_anisotropy_term = np.dot(tan, vec) / (S ** 2)
#             physical_anisotropy_term = -(
#                 (1 / S) * (h1 * (Kt1 / Kn1) + h2 * (Kt2 / Kn2))
#             )
#             cross_diffusion_term = (
#                 mesh_anisotropy_term + physical_anisotropy_term
#             )
#             return cross_diffusion_term
#         if boundary:
#             dot_term = np.dot(-tan, vec) * Kn1
#             cdf_term = h1 * S * Kt1
#             b_cross_difusion_term = (dot_term + cdf_term) / (2 * h1 * S)
#             return b_cross_difusion_term
#
#     # @celery.task
#     def get_nodes_weights(self, method):
#         """Return the node weights."""
#         self.nodes_ws = {}
#         self.nodes_nts = {}
#         # This is the limiting part of the interpoation method. The Dict
#         for node in self.intern_nodes:
#             self.nodes_ws[node] = method(node)
#
#         for node in self.neumann_nodes:
#             self.nodes_ws[node] = method(node, neumann=True)
#             self.nodes_nts[node] = self.nodes_ws[node].pop(node)
#
#     def _node_treatment(self, node, id_left, id_right, K_eq, D_JK=0, D_JI=0.0):
#         """Add flux term from nodes RHS."""
#         RHS = 0.5 * K_eq * (D_JK + D_JI)
#         if node in self.dirichlet_nodes:
#             pressure = self.get_boundary_node_pressure(node)
#             self.Q[id_left] += RHS * pressure
#             self.Q[id_right] += -RHS * pressure
#             # self.Q[id_left[0], 0] += RHS * pressure
#             # self.Q[id_right[0], 0] += - RHS * pressure
#
#         if node in self.intern_nodes:
#             for volume, weight in self.nodes_ws[node].items():
#                 self.ids.append([id_left, id_right])
#                 v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
#                 self.v_ids.append([v_id, v_id])
#                 self.ivalues.append([-RHS * weight, RHS * weight])
#
#         if node in self.neumann_nodes:
#             neu_term = self.nodes_nts[node]
#             self.Q[id_right] += -RHS * neu_term
#             self.Q[id_left] += RHS * neu_term
#             # self.Q[id_right, 0] += - RHS * neu_term
#             # self.Q[id_left, 0] += RHS * neu_term
#
#             for volume, weight_N in self.nodes_ws[node].items():
#                 self.ids.append([id_left, id_right])
#                 v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
#                 self.v_ids.append([v_id, v_id])
#                 self.ivalues.append([-RHS * weight_N, RHS * weight_N])
#
#     def run_solver(self, interpolation_method):
#         """Run solver."""
#         self.get_mobility()
#         self.interpolation_method = interpolation_method
#         t0 = time.time()
#         n_vertex = len(set(self.mesh_data.all_nodes) - self.dirichlet_nodes)
#         print("interpolation runing...")
#         self.get_nodes_weights(interpolation_method)
#         print(
#             "done interpolation...",
#             "took {0} seconds to interpolate over {1} verts".format(
#                 time.time() - t0, n_vertex
#             ),
#         )
#         print("filling the transmissibility matrix...")
#         # begin = time.time()
#
#         try:
#             for volume in self.volumes:
#                 volume_id = self.mb.tag_get_data(self.global_id_tag, volume)[
#                     0
#                 ][0]
#                 RHS = self.mb.tag_get_data(self.source_tag, volume)[0][0]
#                 self.Q[volume_id] += RHS
#                 # self.Q[volume_id, 0] += RHS
#         except Exception:
#             pass
#
#         for face in self.neumann_faces:
#             face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
#                 0
#             ][0]
#             face_flow = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
#             volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
#             volume = np.asarray(volume, dtype="uint64")
#             id_volume = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
#             face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
#             node_crds = self.mb.get_coords(face_nodes).reshape([3, 3])
#             face_area = geo._area_vector(node_crds, norma=True)
#             RHS = face_mobility * face_flow * face_area
#             self.Q[id_volume] += -RHS
#             # self.Q[id_volume, 0] += - RHS
#
#         id_volumes = []
#         all_LHS = []
#         for face in self.dirichlet_faces:
#             face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
#                 0
#             ][0]
#             # '2' argument was initially '0' but it's incorrect
#             I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)
#
#             left_volume = np.asarray(
#                 self.mtu.get_bridge_adjacencies(face, 2, 3), dtype="uint64"
#             )
#             id_volume = self.mb.tag_get_data(self.global_id_tag, left_volume)[
#                 0
#             ][0]
#             id_volumes.append(id_volume)
#
#             JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#             JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
#             LJ = (
#                 self.mb.get_coords([J])
#                 - self.mesh_data.mb.tag_get_data(
#                     self.volume_centre_tag, left_volume
#                 )[0]
#             )
#             N_IJK = np.cross(JI, JK) / 2.0
#             _test = np.dot(LJ, N_IJK)
#             if _test < 0.0:
#                 I, K = K, I
#                 JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#                 JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
#                 N_IJK = np.cross(JI, JK) / 2.0
#             tan_JI = np.cross(N_IJK, JI)
#             tan_JK = np.cross(N_IJK, JK)
#             self.mb.tag_set_data(self.normal_tag, face, N_IJK)
#
#             face_area = np.sqrt(np.dot(N_IJK, N_IJK))
#             h_L = geo.get_height(N_IJK, LJ)
#
#             g_I = self.get_boundary_node_pressure(I)
#             g_J = self.get_boundary_node_pressure(J)
#             g_K = self.get_boundary_node_pressure(K)
#
#             K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
#                 [3, 3]
#             )
#             K_n_L = self.vmv_multiply(N_IJK, face_mobility * K_L, N_IJK)
#             K_L_JI = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JI)
#             K_L_JK = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JK)
#
#             D_JK = self.get_cross_diffusion_term(
#                 tan_JK, LJ, face_area, h_L, K_n_L, K_L_JK, boundary=True
#             )
#             D_JI = self.get_cross_diffusion_term(
#                 tan_JI, LJ, face_area, h_L, K_n_L, K_L_JI, boundary=True
#             )
#             K_eq = (1 / h_L) * (face_area * K_n_L)
#
#             RHS = D_JK * (g_I - g_J) - K_eq * g_J + D_JI * (g_J - g_K)
#             LHS = K_eq
#             all_LHS.append(LHS)
#
#             self.Q[id_volume] += -RHS
#             # self.Q[id_volume, 0] += - RHS
#             # self.mb.tag_set_data(self.flux_info_tag, face,
#             #                      [D_JK, D_JI, K_eq, I, J, K, face_area])
#
#         all_cols = []
#         all_rows = []
#         all_values = []
#         self.ids = []
#         self.v_ids = []
#         self.ivalues = []
#         for face in self.intern_faces:
#             face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
#                 0
#             ][0]
#             left_volume, right_volume = self.mtu.get_bridge_adjacencies(
#                 face, 2, 3
#             )
#             L = self.mesh_data.mb.tag_get_data(
#                 self.volume_centre_tag, left_volume
#             )[0]
#             R = self.mesh_data.mb.tag_get_data(
#                 self.volume_centre_tag, right_volume
#             )[0]
#             dist_LR = R - L
#             I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
#             JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#             JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
#
#             N_IJK = np.cross(JI, JK) / 2.0
#             test = np.dot(N_IJK, dist_LR)
#
#             if test < 0:
#                 left_volume, right_volume = right_volume, left_volume
#                 L = self.mesh_data.mb.tag_get_data(
#                     self.volume_centre_tag, left_volume
#                 )[0]
#                 R = self.mesh_data.mb.tag_get_data(
#                     self.volume_centre_tag, right_volume
#                 )[0]
#                 dist_LR = R - L
#
#             face_area = np.sqrt(np.dot(N_IJK, N_IJK))
#             tan_JI = np.cross(N_IJK, JI)
#             tan_JK = np.cross(N_IJK, JK)
#
#             K_R = self.mb.tag_get_data(self.perm_tag, right_volume).reshape(
#                 [3, 3]
#             )
#             RJ = R - self.mb.get_coords([J])
#             h_R = geo.get_height(N_IJK, RJ)
#
#             K_R_n = self.vmv_multiply(N_IJK, face_mobility * K_R, N_IJK)
#             K_R_JI = self.vmv_multiply(N_IJK, face_mobility * K_R, tan_JI)
#             K_R_JK = self.vmv_multiply(N_IJK, face_mobility * K_R, tan_JK)
#
#             K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
#                 [3, 3]
#             )
#
#             LJ = L - self.mb.get_coords([J])
#             h_L = geo.get_height(N_IJK, LJ)
#
#             K_L_n = self.vmv_multiply(N_IJK, face_mobility * K_L, N_IJK)
#             K_L_JI = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JI)
#             K_L_JK = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JK)
#
#             D_JI = self.get_cross_diffusion_term(
#                 tan_JI,
#                 dist_LR,
#                 face_area,
#                 h_L,
#                 K_L_n,
#                 K_L_JI,
#                 h_R,
#                 K_R_JI,
#                 K_R_n,
#             )
#             D_JK = self.get_cross_diffusion_term(
#                 tan_JK,
#                 dist_LR,
#                 face_area,
#                 h_L,
#                 K_L_n,
#                 K_L_JK,
#                 h_R,
#                 K_R_JK,
#                 K_R_n,
#             )
#
#             K_eq = (K_R_n * K_L_n) / (K_R_n * h_L + K_L_n * h_R) * face_area
#
#             id_right = self.mb.tag_get_data(self.global_id_tag, right_volume)
#             id_left = self.mb.tag_get_data(self.global_id_tag, left_volume)
#
#             col_ids = [id_right, id_right, id_left, id_left]
#             row_ids = [id_right, id_left, id_left, id_right]
#             values = [K_eq, -K_eq, K_eq, -K_eq]
#             all_cols.append(col_ids)
#             all_rows.append(row_ids)
#             all_values.append(values)
#             # wait for interpolation to be done
#             self._node_treatment(I, id_left, id_right, K_eq, D_JK=D_JK)
#             self._node_treatment(
#                 J, id_left, id_right, K_eq, D_JI=D_JI, D_JK=-D_JK
#             )
#             self._node_treatment(K, id_left, id_right, K_eq, D_JI=-D_JI)
#             # self.mb.tag_set_data(self.flux_info_tag, face,
#             #                      [D_JK, D_JI, K_eq, I, J, K, face_area])
#
#         self.T.InsertGlobalValues(self.ids, self.v_ids, self.ivalues)
#         # self.T[
#         #     np.asarray(self.ids)[:, :, 0, 0], np.asarray(self.v_ids)
#         # ] = np.asarray(self.ivalues)
#         self.T.InsertGlobalValues(id_volumes, id_volumes, all_LHS)
#         # self.T[
#         #     np.asarray(id_volumes), np.asarray(id_volumes)
#         # ] = np.asarray(all_LHS)
#         self.T.InsertGlobalValues(all_cols, all_rows, all_values)
#         # self.T[
#         #     np.asarray(all_cols)[:, 0, 0, 0],
#         #     np.asarray(all_rows)[:, 0, 0, 0]
#         # ] = np.asarray(all_values)[:, 0]
#
#         self.T.FillComplete()
#         # M = np.array(
#         #     [
#         #         [self.T[i, j] for i in range(len(self.volumes))]
#         #         for j in range(len(self.volumes))
#         #     ],
#         #     dtype="float64",
#         # )
#         # q = np.asarray(self.Q, dtype="float64")
#         # diagonal_dominance = [
#         #     eval.check_if_matrix_is_diagonal_dominant(i, row)
#         #     for i, row in enumerate(M)
#         # ]
#         # antidiffusive_matrix = [
#         #     eval.check_off_diagonal_nonpositiviness(i, row, q)
#         #     for (i, row), q in zip(enumerate(M), q)
#         # ]
#         mat_fill_time = time.time() - t0
#         print("matrix fill took {0} seconds...".format(mat_fill_time))
#         mesh_size = len(self.volumes)
#         print("running solver...")
#         USE_DIRECT_SOLVER = False
#         linearProblem = Epetra.LinearProblem(self.T, self.x, self.Q)
#         if USE_DIRECT_SOLVER:
#             solver = Amesos.Lapack(linearProblem)
#             print("1) Performing symbolic factorizations...")
#             solver.SymbolicFactorization()
#             print("2) Performing numeric factorizations...")
#             solver.NumericFactorization()
#             print("3) Solving the linear system...")
#             solver.Solve()
#             t = time.time() - t0
#             print(
#                 "Solver took {0} seconds to run over {1} volumes".format(
#                     t, mesh_size
#                 )
#             )
#         else:
#             solver = AztecOO.AztecOO(linearProblem)
#             solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres)
#             # solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_none)
#             solver.SetAztecOption(AztecOO.AZ_pre_calc, AztecOO.AZ_recalc)
#             solver.SetAztecOption(AztecOO.AZ_precond, AztecOO.AZ_dom_decomp)
#             solver.SetAztecOption(AztecOO.AZ_subdomain_solve, AztecOO.AZ_ilu)
#             # solver.SetAztecOption(AztecOO.AZ_kspace, )
#             # solver.SetAztecOption(AztecOO.AZ_orthog, AztecOO.AZ_modified)
#             # solver.SetAztecOption(AztecOO.AZ_conv, AztecOO.AZ_Anorm)
#             solver.Iterate(10000, 1e-5)
#             t = time.time() - t0
#             its = solver.GetAztecStatus()[0]
#             solver_time = solver.GetAztecStatus()[6]
#             import pdb
#
#             pdb.set_trace()
#             self.record_data(f'results_direct.vtk')
#             print(
#                 "Solver took {0} seconds to run over {1} volumes".format(
#                     t, mesh_size
#                 )
#             )
#             print(
#                 "Solver converged at %.dth iteration in %3f seconds."
#                 % (int(its), solver_time)
#             )
#         # self.T = self.T.tocsc()
#         # self.Q = self.Q.tocsc()
#         # self.x = spsolve(self.T, self.Q)
#         # print(np.sum(self.T[50]), self.Q[50])
#         self.mb.tag_set_data(self.pressure_tag, self.volumes, self.x)
#         self.tag_velocity()
