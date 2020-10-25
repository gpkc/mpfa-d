"""This is the begin."""
from pymoab import types
from PyTrilinos import Epetra, AztecOO, Amesos
import solvers.helpers.geometric as geo
import numpy as np
import time
from collections import OrderedDict


# from solvers.helpers import matrix_evaluation as eval

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import find


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
        self.velocity_tag = mesh_data.velocity_tag
        self.node_pressure_tag = mesh_data.node_pressure_tag
        self.face_mobility_tag = mesh_data.face_mobility_tag
        self.left_volume_tag = mesh_data.left_volume_tag

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
        self.volumes = self.mesh_data.all_volumes

        std_map = Epetra.Map(len(self.volumes), 0, self.comm)
        self.T = Epetra.CrsMatrix(Epetra.Copy, std_map, 0)
        self.Q = lil_matrix((len(self.volumes), 1), dtype=np.float)
        if x is None:
            self.x = Epetra.Vector(std_map)
            self.dx = Epetra.Vector(std_map)
        else:
            self.x = x
        # self.T_plus = lil_matrix((len(self.volumes), len(self.volumes)),
        #                     dtype=np.float)
        # self._T = lil_matrix((len(self.volumes), len(self.volumes)),
        #                     dtype=np.float)
        # self._T_plus = lil_matrix((len(self.volumes), len(self.volumes)),
        #                     dtype=np.float)
        # self.T_minus = lil_matrix((len(self.volumes), len(self.volumes)),
        #                     dtype=np.float)
        expanded_matrix = len(self.volumes) + len(boundary_nodes)
        self.T_expanded = lil_matrix(
            (expanded_matrix, expanded_matrix), dtype=np.float
        )
        self.T_plus = lil_matrix(
            (expanded_matrix, expanded_matrix), dtype=np.float
        )
        self.T_plus_aux = lil_matrix(
            (expanded_matrix, expanded_matrix), dtype=np.float
        )
        self.T_minus = lil_matrix(
            (expanded_matrix, expanded_matrix), dtype=np.float
        )
        self.expanded_matrix_node_ids = self.get_node_ids(boundary_nodes)
        self.u = np.zeros(expanded_matrix)

    def get_node_ids(self, nodes):
        max_volumes_ids = max(
            self.mb.tag_get_data(self.global_id_tag, self.volumes)
        )
        return {node: max_volumes_ids + node for node in nodes}

    def record_data(self, file_name):
        """Record data to file."""
        volumes = self.mb.get_entities_by_dimension(0, 3)
        # faces = self.mb.get_entities_by_dimension(0, 2)
        ms = self.mb.create_meshset()
        self.mb.add_entities(ms, volumes)
        # self.mb.add_entities(ms, faces)
        self.mb.write_file(file_name, [ms])

    def tag_verts_pressure(self):
        p_verts = []
        for node in self.mesh_data.all_nodes:
            try:
                p_vert = self.mb.tag_get_data(self.dirichlet_tag, node)
                p_verts.append(p_vert[0])
            except Exception:
                p_vert = 0.0
                p_tag = self.pressure_tag
                nd_weights = self.nodes_ws[node]
                for volume, wt in nd_weights.items():
                    p_vol = self.mb.tag_get_data(p_tag, volume)
                    p_vert += p_vol * wt
                p_verts.append(p_vert)
        self.mb.tag_set_data(
            self.node_pressure_tag, self.mesh_data.all_nodes, p_verts
        )

    def tag_velocity(self):
        self.tag_verts_pressure()
        velocities = []
        for face in self.neumann_faces:
            face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
                0
            ][0]
            face_flow = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
            volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
            volume = np.asarray(volume, dtype="uint64")
            face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
            node_crds = self.mb.get_coords(face_nodes).reshape([3, 3])
            face_area = geo._area_vector(node_crds, norma=True)
            velocity = face_mobility * face_flow * face_area
            velocities.append(velocity)
        self.mb.tag_set_data(self.velocity_tag, self.neumann_faces, velocities)

        velocities = []
        dirichlet_faces = list(self.dirichlet_faces)
        for face in dirichlet_faces:
            face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
                0
            ][0]
            # '2' argument was initially '0' but it's incorrect
            I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)

            left_volume = np.asarray(
                self.mtu.get_bridge_adjacencies(face, 2, 3), dtype="uint64"
            )

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

            g_I = self.mb.tag_get_data(self.node_pressure_tag, I)
            g_J = self.mb.tag_get_data(self.node_pressure_tag, J)
            g_K = self.mb.tag_get_data(self.node_pressure_tag, K)

            K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
                [3, 3]
            )
            K_n_L = self.vmv_multiply(N_IJK, face_mobility * K_L, N_IJK)
            K_L_JI = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JI)
            K_L_JK = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JK)

            D_JK = self.get_cross_diffusion_term(
                tan_JK, LJ, face_area, h_L, K_n_L, K_L_JK, boundary=True
            )
            D_JI = self.get_cross_diffusion_term(
                tan_JI, LJ, face_area, h_L, K_n_L, K_L_JI, boundary=True
            )
            K_eq = (1 / h_L) * (face_area * K_n_L)
            p_vol = self.mb.tag_get_data(self.pressure_tag, left_volume)
            velocity = (
                D_JK * (g_I - g_J) - K_eq * (p_vol - g_J) + D_JI * (g_J - g_K)
            )
            velocities.append(velocity)
        vels = np.asarray(velocities).flatten()
        self.mb.tag_set_data(self.velocity_tag, dirichlet_faces, vels)
        velocities = []
        left_vols = []
        intern_faces = list(self.intern_faces)
        for face in intern_faces:
            face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
                0
            ][0]
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

            K_R_n = self.vmv_multiply(N_IJK, face_mobility * K_R, N_IJK)
            K_R_JI = self.vmv_multiply(N_IJK, face_mobility * K_R, tan_JI)
            K_R_JK = self.vmv_multiply(N_IJK, face_mobility * K_R, tan_JK)

            K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
                [3, 3]
            )

            LJ = L - self.mb.get_coords([J])
            h_L = geo.get_height(N_IJK, LJ)

            K_L_n = self.vmv_multiply(N_IJK, face_mobility * K_L, N_IJK)
            K_L_JI = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JI)
            K_L_JK = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JK)

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
            p_r = self.mb.tag_get_data(self.pressure_tag, right_volume)
            p_l = self.mb.tag_get_data(self.pressure_tag, left_volume)
            p_I = self.mb.tag_get_data(self.node_pressure_tag, I)
            p_J = self.mb.tag_get_data(self.node_pressure_tag, J)
            p_K = self.mb.tag_get_data(self.node_pressure_tag, K)
            velocity = K_eq * (
                p_r - p_l - D_JI * (p_I - p_J) - D_JK * (p_K - p_J)
            )
            velocities.append(velocity)
            left_vols.append(left_volume)
        velocities = np.asarray(velocities).flatten()
        self.mb.tag_set_data(self.velocity_tag, intern_faces, velocities)
        self.mb.tag_set_data(
            self.left_volume_tag, left_vols, np.repeat(1, len(left_vols))
        )

    def get_mobility(self):
        faces = self.mb.get_entities_by_dimension(0, 2)
        try:
            self.mb.tag_get_data(self.face_mobility_tag, faces)
        except RuntimeError:
            mobility_init = np.repeat(1.0, len(faces))
            self.mb.tag_set_data(self.face_mobility_tag, faces, mobility_init)

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
            node_id = self.expanded_matrix_node_ids[node]
            # self.Q[id_left] += RHS * pressure
            # self.Q[id_right] += -RHS * pressure
            self.Q[id_left[0], 0] += RHS * pressure
            self.Q[id_right[0], 0] += -RHS * pressure
            self.T_expanded[id_left[0][0], node_id[0]] -= RHS
            self.T_expanded[id_right[0][0], node_id[0]] += RHS

        if node in self.intern_nodes:
            for volume, weight in self.nodes_ws[node].items():
                self.ids.append([id_left, id_right])
                v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
                self.v_ids.append([v_id, v_id])
                self.ivalues.append([-RHS * weight, RHS * weight])

        if node in self.neumann_nodes:
            neu_term = self.nodes_nts[node]
            # self.Q[id_right] += -RHS * neu_term
            # self.Q[id_left] += RHS * neu_term
            self.Q[id_right, 0] += -RHS * neu_term
            self.Q[id_left, 0] += RHS * neu_term

            for volume, weight_N in self.nodes_ws[node].items():
                self.ids.append([id_left, id_right])
                v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
                self.v_ids.append([v_id, v_id])
                self.ivalues.append([-RHS * weight_N, RHS * weight_N])

    def compute_alpha(
        self, i, j, mi=0.000001
    ):
        print(j, len(self.volumes))
        if j >= len(self.volumes):
            u_i_max = self.u_maxs[i]
            u_i_min = self.u_mins[i]
            u_i = self.u[i]
            u_j = self.u[j]
            if u_i > u_j:
                s_ij = min(2 * mi * (u_i_max - u_i), u_i - u_j)
            else:
                s_ij = max(2 * mi * (u_i_min - u_i), u_i - u_j)
        else:
            u_i_max = self.u_maxs[i]
            u_i_min = self.u_mins[i]
            u_j_min = self.u_mins[j]
            u_i = self.u[i]
            u_j = self.u[j]
            if u_i > u_j:
                s_ij = min(
                    2 * mi * (u_i_max - u_i),
                    u_i - u_j,
                    2 * mi * (u_j - u_j_min),
                )
            else:
                s_ij = max(
                    2 * mi * (u_i_min - u_i),
                    u_i - u_j,
                    2 * mi * (u_j - u_i_max),
                )
        try:
            alpha = s_ij / (u_i - u_j)
        except ZeroDivisionError:
            alpha = 1
        alpha = max(0, alpha)
        return alpha

    def run_solver(self, interpolation_method):
        """Run solver."""
        self.get_mobility()
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
        # begin = time.time()

        try:
            for volume in self.volumes:
                volume_id = self.mb.tag_get_data(self.global_id_tag, volume)[
                    0
                ][0]
                RHS = self.mb.tag_get_data(self.source_tag, volume)[0][0]
                # self.Q[volume_id] += RHS
                self.Q[volume_id, 0] += RHS
        except Exception:
            pass

        for face in self.neumann_faces:
            face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
                0
            ][0]
            face_flow = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
            volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
            volume = np.asarray(volume, dtype="uint64")
            id_volume = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
            face_nodes = self.mtu.get_bridge_adjacencies(face, 0, 0)
            node_crds = self.mb.get_coords(face_nodes).reshape([3, 3])
            face_area = geo._area_vector(node_crds, norma=True)
            RHS = face_mobility * face_flow * face_area
            # self.Q[id_volume] += -RHS
            self.Q[id_volume, 0] += -RHS

        id_volumes = []
        all_LHS = []
        antidiffusive_flux = {}
        for face in self.dirichlet_faces:
            face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
                0
            ][0]
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
            #  calc_grad(I, J, K, vol_left, vol_right=none)
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
            K_n_L = self.vmv_multiply(N_IJK, face_mobility * K_L, N_IJK)
            K_L_JI = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JI)
            K_L_JK = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JK)

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
            if id_volume not in antidiffusive_flux.keys():
                antidiffusive_flux[id_volume] = []
            fluxes = OrderedDict(
                {
                    I: [g_I, D_JK],
                    J: [g_J, -D_JK + D_JI - K_eq],
                    K: [g_K, -D_JI],
                }
            )
            antidiffusive_flux[id_volume].append(fluxes)
            # self.Q[id_volume] += -RHS
            self.Q[id_volume, 0] += -RHS
            # self.mb.tag_set_data(self.flux_info_tag, face,
            #                      [D_JK, D_JI, K_eq, I, J, K, face_area])
            i_id = self.expanded_matrix_node_ids[I]
            j_id = self.expanded_matrix_node_ids[J]
            k_id = self.expanded_matrix_node_ids[K]
            # self.T_expanded[i_id, id_volume] = D_JK
            self.T_expanded[id_volume, i_id] = D_JK
            # self.T_expanded[j_id, id_volume] = -D_JK + D_JI - K_eq
            self.T_expanded[id_volume, j_id] = -D_JK + D_JI - K_eq
            # self.T_expanded[k_id, id_volume] = -D_JI
            self.T_expanded[id_volume, k_id] = -D_JI
            self.T_expanded[i_id, i_id] = 1.0
            self.T_expanded[j_id, j_id] = 1.0
            self.T_expanded[k_id, k_id] = 1.0

        all_cols = []
        all_rows = []
        all_values = []
        self.ids = []
        self.v_ids = []
        self.ivalues = []
        for face in self.intern_faces:
            face_mobility = self.mb.tag_get_data(self.face_mobility_tag, face)[
                0
            ][0]
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

            K_R_n = self.vmv_multiply(N_IJK, face_mobility * K_R, N_IJK)
            K_R_JI = self.vmv_multiply(N_IJK, face_mobility * K_R, tan_JI)
            K_R_JK = self.vmv_multiply(N_IJK, face_mobility * K_R, tan_JK)

            K_L = self.mb.tag_get_data(self.perm_tag, left_volume).reshape(
                [3, 3]
            )

            LJ = L - self.mb.get_coords([J])
            h_L = geo.get_height(N_IJK, LJ)

            K_L_n = self.vmv_multiply(N_IJK, face_mobility * K_L, N_IJK)
            K_L_JI = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JI)
            K_L_JK = self.vmv_multiply(N_IJK, face_mobility * K_L, tan_JK)

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
            self._node_treatment(I, id_left, id_right, K_eq, D_JK=D_JK)
            self._node_treatment(
                J, id_left, id_right, K_eq, D_JI=D_JI, D_JK=-D_JK
            )
            self._node_treatment(K, id_left, id_right, K_eq, D_JI=-D_JI)

        self.T.InsertGlobalValues(self.ids, self.v_ids, self.ivalues)
        self.T.InsertGlobalValues(id_volumes, id_volumes, all_LHS)
        self.T.InsertGlobalValues(all_cols, all_rows, all_values)
        self.T.FillComplete()
        for i in range(len(self.volumes)):
            for j in range(len(self.volumes)):
                if self.T[i][j]:
                    self.T_expanded[i, j] = self.T[i][j]
        _is, js, values = find(self.T_expanded)
        self.T_plus[
            [i for i, j in zip(_is, js) if i != j],
            [j for i, j in zip(_is, js) if i != j],
        ] = np.asarray(
            [max(0, value) for i, j, value in zip(_is, js, values) if i != j]
        )
        self.T_plus[_is, _is] = -self.T_plus[_is].sum(axis=1).transpose()
        self.T_minus = self.T_expanded - self.T_plus
        self.q = self.Q.tocsc()
        self.u[: len(self.volumes)] = spsolve(
            self.T_minus[0: len(self.volumes), 0: len(self.volumes)], self.q
        )

        nodes_pressure = self.mb.tag_get_data(
            self.dirichlet_tag, self.dirichlet_nodes | self.neumann_nodes
        )
        self.u[len(self.volumes):] = nodes_pressure[:, 0]
        residual = -self.T_minus * self.u
        self.T_plus_aux = self.T_plus
        _is, js, _ = find(self.T_plus)
        print("vai entrar no calculo do residuo")
        n = 0
        while max(abs(residual[:len(self.volumes)])) > 1e-3:
            n += 1
            # print("u_i max")
            self.u_maxs = {}
            [
                self.u_maxs.update({
                    i: max(
                            0,
                            self.u[i]
                            + max(
                                [
                                    self.u[j] - self.u[i]
                                    for j in self.mb.tag_get_data(
                                        self.global_id_tag,
                                        self.mtu.get_bridge_adjacencies(
                                            self.volumes[i], 0, 3
                                        )
                                    )
                                    if j != i
                                ]
                            ),
                        )
                    }) for i in range(len(self.volumes))
            ]
            # print("u_i min")
            self.u_mins = {}
            [
                self.u_mins.update({
                    i: min(
                            0,
                            self.u[i]
                            + min(
                                [
                                    self.u[j] - self.u[i]
                                    for j in self.mb.tag_get_data(
                                        self.global_id_tag,
                                        self.mtu.get_bridge_adjacencies(
                                            self.volumes[i], 0, 3))
                                    if j != i
                                ]
                            ),
                        )
                    }) for i in range(len(self.volumes))
            ]
            # print("u_j max")
            # print(f"Solution: u_max: \
            #       {max(self.u[:len(self.volumes)])} \
            #       u_min: {min(self.u[:len(self.volumes)])}")
            # print("calculo de slope limiter")
            rows = []
            cols = []
            import pdb; pdb.set_trace()
            alpha_ijs = []
            [
                (
                    rows.append(i),
                    cols.append(j),
                    alpha_ijs.append(
                        self.compute_alpha(i, j)
                    ),
                ) for i, j in zip(_is, js) if i != j
            ]
            # print(f"max alpha: {max(alpha_ijs)} min alpha: {min(alpha_ijs)}")
            # print("vai recalcular T_plus")
            t0 = time.time()
            new_vals = [
                self.T_plus[i, j] * alpha for i, j, alpha in zip(
                    rows, cols, alpha_ijs
                )
            ]
            self.T_plus[rows, cols] = new_vals
            self.T_plus[
                range(len(self.u)), range(len(self.u))
            ] = self.T_plus.sum(axis=1).transpose() - self.T_plus.diagonal()
            # print("Refaz o assembly da matry com novos fluxos")
            self.T_expanded = self.T_minus + self.T_plus
            # print("calculo do termo fonte")
            _q = (
                -self.T_expanded[: len(self.volumes), len(self.volumes):]
                * nodes_pressure
            )
            # print("calculo de pressao u")
            self.u[: len(self.volumes)] = spsolve(
                self.T_expanded[: len(self.volumes), : len(self.volumes)], _q
            )
            f = self.T_plus * self.u
            a_omega_u = - self.T_minus * self.u
            residual = a_omega_u + f
            print(f"res: {max(abs(residual[:len(self.volumes)]))}")

        self.mb.tag_set_data(self.pressure_tag, self.volumes, self.u[:len(self.volumes)])
        self.record_data(f'results_crr.vtk')
        import pdb

        pdb.set_trace()

        # self.mb.tag_set_data(self.pressure_tag, self.volumes, self.x)

        # self.mb.tag_set_data(self.pressure_tag, self.volumes, self.x)
        # self.tag_velocity()
