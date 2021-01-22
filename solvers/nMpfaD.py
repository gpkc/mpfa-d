# """This is the begin."""
# import pdb
#
# from pymoab import types
# from PyTrilinos import Epetra, AztecOO, Amesos
# import solvers.helpers.geometric as geo
# import numpy as np
# import time
# from collections import OrderedDict
#
# s_ij = lambda my_list: min(my_list) \
#     if all([i > 0 for i in my_list]) \
#     else (max(my_list) if all([i < 0 for i in my_list]) else 0)
# # from solvers.helpers import matrix_evaluation as eval
#
# from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve
# from scipy.sparse import find
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
#
#         self.source_tag = mesh_data.source_tag
#         self.global_id_tag = mesh_data.global_id_tag
#         self.volume_centre_tag = mesh_data.volume_centre_tag
#
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
#         # Removes vertices that have both Neumann and Dirichlet tag, but prior Dirichlet tags
#         self.neumann_nodes = self.neumann_nodes - self.dirichlet_nodes
#         boundary_nodes = self.dirichlet_nodes | self.neumann_nodes
#         self.intern_nodes = set(self.mesh_data.all_nodes) - boundary_nodes
#
#         self.dirichlet_faces = mesh_data.dirichlet_faces
#         self.neumann_faces = mesh_data.neumann_faces
#         self.intern_faces = mesh_data.intern_faces()
#         self.volumes = self.mesh_data.all_volumes
#         self.vol_volumes = {}
#         [self.get_volume(a_volume) for a_volume in self.volumes]
#         std_map = Epetra.Map(len(self.volumes), 0, self.comm)
#         self.T = Epetra.CrsMatrix(Epetra.Copy, std_map, 0)
#         self.Q = lil_matrix((len(self.volumes), 1), dtype=np.float)
#         self._Q = Epetra.Vector(std_map)
#         if x is None:
#             self.x = Epetra.Vector(std_map)
#             self.dx = Epetra.Vector(std_map)
#         else:
#             self.x = x
#         expanded_matrix = len(self.volumes) + len(boundary_nodes)
#         self.T_expanded = lil_matrix(
#             (expanded_matrix, expanded_matrix), dtype=np.float
#         )
#         self.T_plus = lil_matrix(
#             (expanded_matrix, expanded_matrix), dtype=np.float
#         )
#         self.T_plus_aux = lil_matrix(
#             (expanded_matrix, expanded_matrix), dtype=np.float
#         )
#         self.T_minus = lil_matrix(
#             (expanded_matrix, expanded_matrix), dtype=np.float
#         )
#         self.expanded_matrix_node_ids = self.get_node_ids(boundary_nodes)
#         self.u = np.zeros(expanded_matrix)
#
#     def get_node_ids(self, nodes):
#         max_volumes_ids = max(
#             self.mb.tag_get_data(self.global_id_tag, self.volumes)
#         )
#         return {
#             node: max_volumes_ids + index + 1 for index, node in enumerate(nodes)
#         }
#
#     def get_volume(self, a_volume):
#         vol_nodes = self.mtu.get_bridge_adjacencies(a_volume, 0, 0)
#         vol_crds = self.mb.get_coords(vol_nodes)
#         vol_crds = np.reshape(vol_crds, ([4, 3]))
#         _id = self.mb.tag_get_data(self.global_id_tag, a_volume)
#         volume = self.mesh_data.get_tetra_volume(vol_crds)
#         self.vol_volumes.update({_id[0][0]: volume})
#
#     def compute_b_parameter(self, u_j, u_vol, u_max, u_min):
#         if u_j == u_vol:
#             val = 1
#         if u_j > u_vol:
#             val = (u_max - u_vol) / (u_j - u_vol)
#         else:
#             val = (u_min - u_vol) / (u_j - u_vol)
#         b = min(1, val)
#         return b
#
#     def get_grad(self, a_volume):
#         vol_faces = self.mtu.get_bridge_adjacencies(a_volume, 2, 2)
#         vol_nodes = self.mtu.get_bridge_adjacencies(a_volume, 0, 0)
#         vol_crds = self.mb.get_coords(vol_nodes)
#         vol_crds = np.reshape(vol_crds, ([4, 3]))
#         vol_volume = self.mesh_data.get_tetra_volume(vol_crds)
#         I, J, K = self.mtu.get_bridge_adjacencies(vol_faces[0], 2, 0)
#         L = list(set(vol_nodes).difference(set(self.mtu.get_bridge_adjacencies(vol_faces[0], 2, 0))))
#         JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#         JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
#         LJ = self.mb.get_coords([J]) - self.mb.get_coords(L)
#         N_IJK = np.cross(JI, JK) / 2.0
#
#         test = np.dot(LJ, N_IJK)
#         if test < 0.0:
#             I, K = K, I
#             JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#             JK = self.mb.get_coords([K]) - self.mb.get_coords(
#                 [J]
#             )
#             N_IJK = np.cross(JI, JK) / 2.0
#
#         # tan_JI = np.cross(JI, N_IJK)
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
#         grad_p = -(1 / (6 * vol_volume)) * (
#             grad_normal + grad_cross_I + grad_cross_K
#         )
#         return grad_p
#
#
#     def compute_mi(self, a_volume):
#         vol_faces = self.mtu.get_bridge_adjacencies(a_volume, 2, 2)
#         vol_nodes = self.mtu.get_bridge_adjacencies(a_volume, 0, 0)
#         vol_crds = self.mb.get_coords(vol_nodes)
#         vol_crds = np.reshape(vol_crds, ([4, 3]))
#         vol_volume = self.mesh_data.get_tetra_volume(vol_crds)
#         vol_centre = self.mb.get_coords(a_volume)
#         mi = []
#         for face in vol_faces:
#             I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)
#             L = list(set(vol_nodes).difference(
#                 set(self.mtu.get_bridge_adjacencies(face, 2, 0)))
#             )
#             JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#             JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
#             LJ = self.mb.get_coords([J]) - self.mb.get_coords(L)
#             N_IJK = np.cross(JI, JK) / 2.0
#             test = np.dot(LJ, N_IJK)
#             if test < 0.0:
#                 I, K = K, I
#                 JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
#                 JK = self.mb.get_coords([K]) - self.mb.get_coords(
#                     [J]
#                 )
#                 N_IJK = np.cross(JI, JK) / 2.0
#             adj_vol = set(self.mtu.get_bridge_adjacencies(face, 2, 3))\
#                 .difference(set([a_volume, ]))
#             if not adj_vol:
#                 continue
#             adj_vol_centre = self.mb.get_coords(adj_vol)
#             r = vol_centre - adj_vol_centre
#             dot_product = np.dot(N_IJK, r)
#             mi.append(abs(dot_product))
#         mi = sum(mi) / vol_volume
#         return mi
#
#     def compute_all_mi(self):
#         mis = {volume: self.compute_mi(volume) for volume in self.volumes}
#         return mis
#
#
#
#
#
#
#
#
#     def compute_slip_fact(self):
#         mis = self.compute_all_mi()
#         rows = []
#         cols = []
#         for face in self.dirichlet_faces | self.neumann_faces:
#             volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
#             u = self.mb.tag_get_data(self.pressure_tag, volume)
#             max_v, min_v = self.get_local_min_max(volume)
#             verts = self.mtu.get_bridge_adjacencies(face, 2, 0)
#             avg_verts_pressure = self.mb.tag_get_data(
#                 self.node_pressure_tag, verts
#             ).mean()
#             lp = u - avg_verts_pressure
#             volume = np.asarray(volume)[0]
#             if u > avg_verts_pressure:
#                 lfs = 2 * mis[volume] * (max_v - u)
#                 s = min(lfs, lp)
#             else:
#                 rts = 2 * mis[volume] * (min_v - u)
#                 s = max(lp, rts)
#             print(f"At boundary: {(s + 1e-20) / (lp + 1e-20)}")
#         for face in self.intern_faces:
#             volume, adj = self.mtu.get_bridge_adjacencies(face, 2, 3)
#             u = self.mb.tag_get_data(self.pressure_tag, volume)
#             max_v, min_v = self.get_local_min_max(volume)
#             max_v_adj, min_v_adj = self.get_local_min_max(adj)
#             u_adj = self.mb.tag_get_data(self.pressure_tag, adj)
#             lp = u - u_adj
#             if u > u_adj:
#                 lfs = 2 * mis[volume] * (max_v - u)
#                 rts = 2 * mis[adj] * (u_adj - min_v_adj)
#                 s = min(lfs, lp, rts)
#             else:
#                 lfs = 2 * mis[volume] * (min_v - u)
#                 rts = 2 * mis[adj] * (u_adj - max_v)
#                 s = max(lfs, lp, rts)
#             print(f"At inner: {(s + 1e-20) / (lp + 1e-20)}")
#
#     def compute_slip_factor(self):
#         mis = self.compute_all_mi()
#         for volume in self.volumes:
#             u = self.mb.tag_get_data(self.pressure_tag, volume)
#             adjs = self.mtu.get_bridge_adjacencies(volume, 2, 3)
#             max_v, min_v = self.get_local_min_max(volume)
#             # if len(adjs) == 4:
#             for adj in adjs:
#                 max_v_adj, min_v_adj = self.get_local_min_max(adj)
#                 u_adj = self.mb.tag_get_data(self.pressure_tag, adj)
#                 lp = u - u_adj
#                 if u > u_adj:
#                     lfs = 2 * mis[volume] * (max_v - u)
#                     rts = 2 * mis[adj] * (u_adj - min_v_adj)
#                     s = min(lfs, lp, rts)
#                 else:
#                     lfs = 2 * mis[volume] * (min_v - u)
#                     rts = 2 * mis[adj] * (u_adj - max_v)
#                     s = max(lfs, lp, rts)
#                 print ((s + 1e-20) / (lp + 1e-20))
#                 if len(adjs) < 4:
#                     print('this is boundary vol')
#
#
#     def compute_vert_pressure(self, node):
#         all_vols = [
#             self.mb.tag_get_data(
#                 self.global_id_tag, self.nodes_ws[node].keys()
#             ) for node in self.nodes_ws.keys()
#         ]
#         its = set()
#         [[its.add(i) for i in item] for item in all_vols]
#         volumes_pressure = self.u[
#             self.mb.tag_get_data(
#                 self.global_id_tag, self.nodes_ws[node].keys()
#             )][:, 0]
#         volumes_ids = self.mb.tag_get_data(
#             self.global_id_tag, self.nodes_ws[node].keys()
#         )
#         max_pressure = max(volumes_pressure)
#         min_pressure = min(volumes_pressure)
#         pressure = np.dot(
#             self.u[
#                 self.mb.tag_get_data(
#                     self.global_id_tag, self.nodes_ws[node].keys()
#                 )
#             ][:, 0],
#             np.asarray([value for value in self.nodes_ws[node].values()])
#         )
#         _bs = [{key: self.compute_b_parameter(pressure, u_vol, max_pressure, min_pressure)} for key, u_vol in zip(volumes_ids[:, 0], volumes_pressure)]
#         # bs = {}
#         bs = {
#             key: self.compute_b_parameter(
#                 pressure, u_vol, max_pressure, min_pressure
#             ) for key, u_vol in zip(volumes_ids[:, 0], volumes_pressure)
#         }
#         # import pdb; pdb.set_trace()
#         # data = {"pressure": pressure, "max": max_pressure, "min": min_pressure}
#         # import pdb; pdb.set_trace()
#         return bs
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
    # def tag_verts_pressure(self):
    #     print("Will tag vertices pressure")
    #     p_verts = []
    #     for node in self.mesh_data.all_nodes:
    #         try:
    #             p_vert = self.mb.tag_get_data(self.dirichlet_tag, node)
    #             p_verts.append(p_vert[0])
    #         except Exception:
    #             p_vert = 0.0
    #             p_tag = self.pressure_tag
    #             nd_weights = self.nodes_ws[node]
    #             for volume, wt in nd_weights.items():
    #                 p_vol = self.mb.tag_get_data(p_tag, volume)
    #                 p_vert += p_vol * wt
    #             p_verts.append(p_vert)
    #     self.mb.tag_set_data(
    #         self.node_pressure_tag, self.mesh_data.all_nodes, p_verts
    #     )
    #     print("Done tagging vertices pressure!!")
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
#             face_mobility = self.mb.tag_get_data(
#                 self.face_mobility_tag, face
#             )[0][0]
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
#             # tan_JI = np.cross(JI, N_IJK)
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
#             K_L = self.mb.tag_get_data(
#                 self.perm_tag, left_volume
#             ).reshape([3, 3])
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
#             # tan_JI = np.cross(JI, N_IJK)
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
#             node_id = self.expanded_matrix_node_ids[node]
#             self._Q[id_left] += RHS * pressure
#             self._Q[id_right] += -RHS * pressure
#             self.Q[id_left[0], 0] += RHS * pressure
#             self.Q[id_right[0], 0] += -RHS * pressure
#             self.T_expanded[id_left[0][0], node_id[0]] -= RHS
#             self.T_expanded[id_right[0][0], node_id[0]] += RHS
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
#             self._Q[id_right] += -RHS * neu_term
#             self._Q[id_left] += RHS * neu_term
#             self.Q[id_right, 0] += -RHS * neu_term
#             self.Q[id_left, 0] += RHS * neu_term
#
#             for volume, weight_N in self.nodes_ws[node].items():
#                 self.ids.append([id_left, id_right])
#                 v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
#                 self.v_ids.append([v_id, v_id])
#                 self.ivalues.append([-RHS * weight_N, RHS * weight_N])
#
#     def compute_alpha(
#         self, i, _j, mi=None
#     ):
#         if _j >= len(self.volumes):
#             # _j = j
#             idx = list(self.expanded_matrix_node_ids.values()).index(_j)
#             val = list(self.expanded_matrix_node_ids.keys())[idx]
#             adj_vols = self.mb.tag_get_data(
#                 self.global_id_tag, self.mtu.get_bridge_adjacencies(val, 0, 3)
#             )
#             grad_i = self.grads.get(i).get("grad")
#             coords_i = self.mb.get_coords(self.grads.get(i).get('vol'))
#             # grads = np.asarray(
#             #     [self.grads.get(vol[0]).get('grad')[0] for vol in adj_vols]
#             # )
#             dists = np.asarray([
#                 coords_i - self.mb.get_coords(
#                     self.grads.get(vol[0]).get('vol')
#                 )
#                 for vol in adj_vols
#             ])
#             volumes = np.asarray(
#                 [self.vol_volumes.get(vol[0]) for vol in adj_vols]
#             )
#             avg_dist = np.asarray(
#                 [sum(volumes * dists[:, i]) / sum(volumes) for i in range(3)]
#             )
#             mi = np.dot(avg_dist - coords_i, grad_i[0])
#             # mi = np.dot(
#             #     np.asarray(
#             #         [np.dot(grad, dist) for grad, dist in zip(grads, dists)]
#             #     ), volumes
#             # )
#             # get averaged grad from other volumes
#             u_i_max = self.u_maxs[i]
#             u_i_min = self.u_mins[i]
#             u_i = self.u[i]
#             u_j = self.u[_j]
#             if u_i > u_j:
#                 s_ij = min(2 * mi * (u_i_max - u_i), u_i - u_j)
#             else:
#                 s_ij = max(2 * mi * (u_i_min - u_i), u_i - u_j)
#         else:
#             grad_i = self.grads.get(i).get("grad")
#             coords_i = self.mb.get_coords(self.grads.get(i).get('vol'))
#             grad_j = self.grads.get(_j).get("grad")
#             coords_j = self.mb.get_coords(self.grads.get(_j).get('vol'))
#             u_i_max = self.u_maxs[i]
#             u_i_min = self.u_mins[i]
#             u_j_min = self.u_mins[_j]
#             u_i = self.u[i]
#             u_j = self.u[_j]
#             mi_ij = np.dot(
#                 grad_i, (coords_i - coords_j)
#             )
#             mi_ji = np.dot(
#                 grad_j, (coords_j - coords_i)
#             )
#             if u_i > u_j:
#                 s_ij = min(
#                     2 * mi_ij * (u_i_max - u_i),
#                     u_i - u_j,
#                     2 * mi_ji * (u_j - u_j_min),
#                 )
#             else:
#                 s_ij = max(
#                     2 * mi_ij * (u_i_min - u_i),
#                     u_i - u_j,
#                     2 * mi_ji * (u_j - u_i_max),
#                 )
#         # try:
#         #     alpha = (s_ij) / (u_i - u_j)
#         # except ZeroDivisionError:
#         alpha = (s_ij + 1e-20) / (u_i - u_j + 1e-20)
#         # TODO: testar abrir a malha com apenas alpha na iteração
#
#         # alpha = 1
#
#         return
#
#
#
#
    # def get_global_rows(self):
    #     aux_list = []
    #     epetra_global_rows = np.asarray(
    #         [
    #             self.T.ExtractGlobalRowCopy(row)[1]
    #             for row in range(len(self.volumes))]
    #     )
    #     for row, arrs in enumerate(epetra_global_rows):
    #         for val in arrs:
    #             aux_list.append([row, val])
    #     rows = np.asarray(aux_list)[:, 0]
    #     cols = np.asarray(aux_list)[:, 1]
    #     return rows, cols
#
#     @staticmethod
#     def create_sp_matrix(shape, dtype=np.float64):
#         return lil_matrix((shape, shape), dtype=dtype)
#
#     def copy_mat(self, rows, cols, shape=None, dtype=np.float64):
#         A = self.create_sp_matrix(shape, dtype)
#         A[rows, cols] = [self.T[row][col] for row, col in zip(rows, cols)]
#         return A
#
#     def solve_original_problem(self, mat: lil_matrix, q: np.array):
#
#         if any([mat.shape[0] < len(self.volumes), len(q) < len(self.volumes)]):
#             raise (
#                 "not solvable. matrix length must be at leat the size of the original problem"
#             )
#
#         x = spsolve(
#             mat[:len(self.volumes), :len(self.volumes)].tocsc(), q[:len(self.volumes)]
#         )
#         return x
#
#     def get_off_diagonal_index(self, rows_and_cols):
#         zipped = zip(rows_and_cols[0], rows_and_cols[1])
#         off_diagonal_indexes = [
#             first for first, second in zipped if first != second
#         ]
#         return off_diagonal_indexes
#
#     def only_positive_values(self, rows, cols):
#         positive_values = np.asarray(
#             [
#                 [row, col] for row, col
#                 in zip(rows, cols) if self.T[row, col] > 0
#             ]
#         )
#
#         positive_rows = positive_values[:, 0]
#         positive_cols = positive_values[:, 1]
#         return positive_rows, positive_cols
#
#     def sum_into_diagonal(self, mat: lil_matrix):
#         size = mat.shape[0]
#         diagonals = [-np.sum(mat[row].toarray()) for row in range(mat.shape[0])]
#         mat[range(size), range(size)] = diagonals
#         mat.tocsc()
#
#
#     def get_local_min_max(self, volume):
#         volume_pressure = self.mb.tag_get_data(self.pressure_tag, volume).flatten()
#         adj_volumes = self.mtu.get_bridge_adjacencies(volume, 2, 3)
#         volume_vertices = self.mtu.get_bridge_adjacencies(volume, 0, 0)
#         adj_volumes_pressure = self.mb.tag_get_data(self.pressure_tag, adj_volumes)
#         adj_volumes_verts = [
#             self.mtu.get_bridge_adjacencies(adj_vol, 0, 0)
#             for adj_vol in adj_volumes
#         ]
#         all_volumes_verts = set(
#             np.asarray(adj_volumes_verts).flatten()
#         ).union(set(np.asarray(volume_vertices)))
#         all_vertices_in_patch_pressure = self.mb.tag_get_data(
#             self.node_pressure_tag,
#             np.asarray(list(all_volumes_verts), dtype=np.uint64)
#         ).flatten()
#         surrounding_pressure_patch = all_vertices_in_patch_pressure\
#                                          .flatten().tolist()\
#                                      + adj_volumes_pressure.tolist()
#         p_max_local = max(
#             0, volume_pressure + max(
#                 [adj_p - volume_pressure for adj_p in surrounding_pressure_patch]
#             )
#         )
#         p_min_local = min(
#             0, volume_pressure + min(
#                 [adj_p - volume_pressure for adj_p in surrounding_pressure_patch]
#             )
#         )
#         return p_max_local, p_min_local
#
#     def get_local_maxima_and_minima(self, vol):
#         adjacent_vols = self.mtu.get_bridge_adjacencies(vol, 2, 3)
#         adj_vols = [*adjacent_vols]
#         vol_press = self.mb.tag_get_data(self.pressure_tag, vol)
#         vols_pressure = self.mb.tag_get_data(self.pressure_tag, adj_vols)
#         deltas = []
#         for press in vols_pressure:
#             deltas.append(press - vol_press)
#         if len(adjacent_vols) < 4:
#             vol_verts = self.mtu.get_bridge_adjacencies(vol, 3, 0)
#             dirichlet_verts_only = list(set(vol_verts).intersection(self.dirichlet_nodes))
#             dirichlet_nodes_pressure = [
#                 self.get_boundary_node_pressure(dirichlet_node)
#                 for dirichlet_node in dirichlet_verts_only
#             ]
#             for press in dirichlet_nodes_pressure:
#                 deltas.append(press - vol_press)
#         local_maxima = max(0, vol_press + max(deltas))
#         local_minima = min(0, vol_press + min(deltas))
#         return local_maxima, local_minima
#
#     def get_slope_limiter(self, vol):
#         pdb.set_trace()
#         vol_press = self.mb.tag_get_data(self.pressure_tag, vol)
#         vol_coords = self.mb.get_coords(vol)
#         vol_local_maxima, vol_local_minima = self.get_local_maxima_and_minima(vol)
#         vol_id = self.mb.tag_get_data(self.global_id_tag, vol)
#         adjacent_vols = self.mtu.get_bridge_adjacencies(vol, 2, 3)
#         grad = self.get_grad(vol)
#         slopes = []
#         for adj_vol in adjacent_vols:
#             adj_vol_id = self.mb.tag_get_data(self.global_id_tag, adj_vol)
#             adj_vol_local_maxima, adj_vol_local_minima = self.get_local_maxima_and_minima(adj_vol)
#             adj_vol_press = self.mb.tag_get_data(self.pressure_tag, adj_vol)
#             adj_vol_grad = self.get_grad(adj_vol)
#             adj_vol_coords = self.mb.get_coords(adj_vol)
#             r = vol_coords = adj_vol_coords
#             mi_ij = np.dot(grad, r)
#             mi_ji = np.dot(adj_vol_grad, -r)
#             lp = vol_press - adj_vol_press
#             if vol_press > adj_vol_press:
#                 l_side = (vol_local_maxima - vol_press)
#                 r_side = (adj_vol_press - adj_vol_local_minima)
#                 s_ij = min(l_side, lp, r_side)
#             else:
#                 l_side = (vol_local_minima - vol_press)
#                 r_side = (adj_vol_press - vol_local_maxima)
#                 s_ij = max(l_side, -lp, r_side)
#                 lp = -lp
#             limiter = (s_ij + 1e-20) / (lp + 1e-20)
#             slopes.append(limiter)
#         return slopes
#
#     # **************
#     # def calculate_alpha(self):
#     #     for vol in self.volumes:
#     #         vol_press = self.mb.tag_get_data(self.pressure_tag, vol)
#     #         # vol_coords = solver.mb.get_coords(vol)
#     #         vol_local_maxima, vol_local_minima = self.get_local_maxima_and_minima(vol)
#     #         vol_id = self.mb.tag_get_data(self.global_id_tag, vol)
#     #         adjacent_vols = self.mtu.get_bridge_adjacencies(vol, 2, 3)
#     #         for adj_vol in adjacent_vols:
#     #             adj_vol_id = self.mb.tag_get_data(self.global_id_tag, adj_vol)
#     #             adj_vol_local_maxima, adj_vol_local_minima = self.get_local_maxima_and_minima(adj_vol)
#     #             adj_vol_press = self.mb.tag_get_data(self.pressure_tag, adj_vol)
#     #             # adj_vol_grad = solver.get_grad(adj_vol)
#     #             # adj_vol_coords = solver.mb.get_coords(adj_vol)
#     #             # r = vol_coords = adj_vol_coords
#     #             # mi_ij = np.dot(grad, r)
#     #             # mi_ji = np.dot(adj_vol_grad, -r)
#     #             lp = vol_press - adj_vol_press
#     #             if vol_press > adj_vol_press:
#     #                 l_side = (vol_local_maxima - vol_press)
#     #                 r_side = (adj_vol_press - adj_vol_local_minima)
#     #                 s_ij = min(l_side, lp, r_side)
#     #             else:
#     #                 l_side = (vol_local_minima - vol_press)
#     #                 r_side = (adj_vol_press - vol_local_maxima)
#     #                 s_ij = max(l_side, -lp, r_side)
#     #                 lp = -lp
#     #             limiter = (s_ij + 1e-20) / (lp + 1e-20)
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
#                 self._Q[volume_id] += RHS
#                 self.Q[volume_id, 0] += RHS
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
#             self._Q[id_volume] += -RHS
#             self.Q[id_volume, 0] += -RHS
#
#         id_volumes = []
#         all_LHS = []
#         antidiffusive_flux = {}
#         for face in self.dirichlet_faces:
#             face_mobility = self.mb.tag_get_data(
#                 self.face_mobility_tag, face
#             )[0][0]
#             # '2' argument was initially '0' but it's incorrect
#             I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)
#
#             left_volume = np.asarray(
#                 self.mtu.get_bridge_adjacencies(face, 2, 3), dtype="uint64"
#             )
#             id_volume = self.mb.tag_get_data(
#                 self.global_id_tag, left_volume
#             )[0][0]
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
#             #  calc_grad(I, J, K, vol_left, vol_right=none)
#             tan_JI = np.cross(JI, N_IJK)
#             # tan_JI = np.cross(N_IJK, JI)
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
#             if id_volume not in antidiffusive_flux.keys():
#                 antidiffusive_flux[id_volume] = []
#             fluxes = OrderedDict(
#                 {
#                     I: [g_I, D_JK],
#                     J: [g_J, -D_JK + D_JI - K_eq],
#                     K: [g_K, -D_JI],
#                 }
#             )
#             antidiffusive_flux[id_volume].append(fluxes)
#             self._Q[id_volume] += -RHS
#             self.Q[id_volume, 0] += -RHS
#             # self.T_expanded[:len(self.volumes), len(self.volumes):len(self.u)] * self.u[len(self.volumes):]
#             # self.mb.tag_set_data(self.flux_info_tag, face,
#             #                      [D_JK, D_JI, K_eq, I, J, K, face_area])
#             i_id = self.expanded_matrix_node_ids[I]
#             j_id = self.expanded_matrix_node_ids[J]
#             k_id = self.expanded_matrix_node_ids[K]
#             # self.T_expanded[i_id, id_volume] = D_JK
#             self.T_expanded[id_volume, i_id] = D_JK
#             # self.T_expanded[j_id, id_volume] = -D_JK + D_JI - K_eq
#             self.T_expanded[id_volume, j_id] = -D_JK + D_JI - K_eq
#             # self.T_expanded[k_id, id_volume] = -D_JI
#             self.T_expanded[id_volume, k_id] = -D_JI
#             self.T_expanded[i_id, i_id] = 1.0
#             self.T_expanded[j_id, j_id] = 1.0
#             self.T_expanded[k_id, k_id] = 1.0
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
#             # tan_JI = np.cross(JI, N_IJK)
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
#             self._node_treatment(I, id_left, id_right, K_eq, D_JK=D_JK)
#             self._node_treatment(
#                 J, id_left, id_right, K_eq, D_JI=D_JI, D_JK=-D_JK
#             )
#             self._node_treatment(K, id_left, id_right, K_eq, D_JI=-D_JI)
#
#         self.T.InsertGlobalValues(self.ids, self.v_ids, self.ivalues)
#         self.T.InsertGlobalValues(id_volumes, id_volumes, all_LHS)
#         self.T.InsertGlobalValues(all_cols, all_rows, all_values)
#         self.id_volumes = id_volumes
#         self.all_cols = all_cols
#         self.all_rows = all_rows
#         self.T.FillComplete()
#
#
#         # mesh_size = len(self.volumes)
#         # print("running solver...")
#         # USE_DIRECT_SOLVER = False
#         # linearProblem = Epetra.LinearProblem(self.T, self.x, self._Q)
#         # if USE_DIRECT_SOLVER:
#         #     solver = Amesos.Lapack(linearProblem)
#         #     print("1) Performing symbolic factorizations...")
#         #     solver.SymbolicFactorization()
#         #     print("2) Performing numeric factorizations...")
#         #     solver.NumericFactorization()
#         #     print("3) Solving the linear system...")
#         #     solver.Solve()
#         #     t = time.time() - t0
#         #     print(
#         #         "Solver took {0} seconds to run over {1} volumes".format(
#         #             t, mesh_size
#         #         )
#         #     )
#         # else:
#         #     solver = AztecOO.AztecOO(linearProblem)
#         #     solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_gmres)
#         #     solver.SetAztecOption(AztecOO.AZ_output, AztecOO.AZ_none)
#         #     solver.SetAztecOption(AztecOO.AZ_precond, AztecOO.AZ_Jacobi)
#         #     solver.SetAztecOption(AztecOO.AZ_kspace, 1251)
#         #     solver.SetAztecOption(AztecOO.AZ_orthog, AztecOO.AZ_modified)
#         #     solver.SetAztecOption(AztecOO.AZ_conv, AztecOO.AZ_Anorm)
#         #     solver.Iterate(8000, 1e-10)
#         #     t = time.time() - t0
#         #     its = solver.GetAztecStatus()[0]
#         #     solver_time = solver.GetAztecStatus()[6]
#         #     print(
#         #         "Solver took {0} seconds to run over {1} volumes".format(
#         #             t, mesh_size
#         #         )
#         #     )
#         #     print(
#         #         "Solver converged at %.dth iteration in %3f seconds."
#         #         % (int(its), solver_time)
#         #     )
#         # self.mb.tag_set_data(self.pressure_tag, self.volumes, self.x)
#         # coords_x = [
#         #     coords[0] for coords in self.mb.get_coords(
#         #         self.volumes
#         #     ).reshape([len(self.volumes), 3])
#         # ]
#         # with open('nresults.csv', 'w') as results:
#         #     for coord, x in zip(coords_x, self.x):
#         #         results.write(f"{str(coord)};{str(x)}\n")
#         # print("asdfasdlkforgnadmfbnsorjgalçskdfnsdçfgm\nasdkjfhaskldfhaklsdjh")
#         # import pdb
#         # pdb.set_trace()
#         # for i in range(len(self.volumes)):
#         #     for j in range(len(self.volumes)):
#         #         if self.T[i][j]:
#         #             self.T_expanded[i, j] = self.T[i][j]
#         # _is, js, values = find(self.T_expanded)
#         # self.q = np.asarray(self._Q)
#         # self.T_expanded.tocsc()
#         # self.u[: len(self.volumes)] = spsolve(
#         #     self.T_expanded[0: len(self.volumes), 0: len(self.volumes)],
#         #     self.q[0: len(self.volumes)]
#         # )
#         # coords_x = [
#         #     coords[0] for coords in self.mb.get_coords(
#         #         self.volumes
#         #     ).reshape([len(self.volumes), 3])
#         # ]
#         # with open('n_t_expanded_results.csv', 'w') as results:
#         #     for coord, x in zip(coords_x, self.u[: len(self.volumes)]):
#         #         results.write(f"{str(coord)};{str(x)}\n")
#         # self.T_plus[
#         #     [i for i, j in zip(_is, js) if i != j],
#         #     [j for i, j in zip(_is, js) if i != j],
#         # ] = np.asarray(
#         #     [max(0, value) for i, j, value in zip(_is, js, values) if i != j]
#         # )
#         # self.T_plus[_is, _is] = -self.T_plus[_is].sum(axis=1).transpose()
#         # self.T_minus = self.T_expanded - self.T_plus
#         # nodes_pressure = self.mb.tag_get_data(
#         #     self.dirichlet_tag, self.dirichlet_nodes
#         # )
#         # self.u[len(self.volumes):] = nodes_pressure[:, 0]
#         # residual = self._Q - (
#         #         (self.T_minus + self.T_plus)[:len(self.volumes), :len(self.volumes)] * self.u[:len(self.volumes)]
#         # )
#         # residual = np.asarray(residual).flatten()
#         # print("vai entrar no calculo do residuo")
#         # n = 0
#         # self.mb.tag_set_data(
#         #     self.pressure_tag, self.volumes, self.u[:len(self.volumes)]
#         # )
#         # vols = [
#         #     self.vol_volumes.get(vol[0]) for vol in self.mb.tag_get_data(
#         #         self.global_id_tag, self.volumes
#         #     )
#         # ]
#         # tol = np.dot(vols, abs(residual[:len(self.volumes)]))
#         # while np.average(abs(residual[:len(self.volumes)])) > 1e-3:
#         # myCounter = 0
#         # while tol > 1e-3:
#         #     _is, js, _ = find(self.T_plus)
#         #     self.tag_verts_pressure()
#         #     # its = list(
#         #     #     map(self.compute_vert_pressure, self.nodes_ws.keys())
#         #     # )
#         #     # import pdb; pdb.set_trace()
#         #     # bs = {}
#         #     # for item in its:
#         #     #     for key, value in item.items():
#         #     #         if bs.get(key):
#         #     #             bs[key] = min(bs.get(key), value)
#         #     #         else:
#         #     #             pass
#         #     # limiter = {key}
#         #     # limiters = {
#         #     #     key: value for key, value in list(
#         #     #         map(self.compute_vert_pressure, self.nodes_ws.keys())
#         #     #     )
#         #     # }
#         #     self.grads = {}
#         #     for volume in self.volumes:
#         #         vol_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
#         #         grad = self.get_grad(volume)
#         #         self.grads.update({vol_id: {"grad": grad, "vol": volume}})
#         #     n += 1
#         #     self.max_min = {}
#         #     self.u_maxs = {}
#         #     self.u_mins = {}
#         #     for volume in self.volumes:
#         #         adj_volumes = self.mtu.get_bridge_adjacencies(
#         #             volume, 2, 3
#         #         )
#         #         volumes_in_patch_ids = self.mb.tag_get_data(
#         #             self.global_id_tag, adj_volumes
#         #         )
#         #         volume_id = self.mb.tag_get_data(
#         #             self.global_id_tag, volume
#         #         )
#         #         neigh_pressure = self.u[volumes_in_patch_ids].flatten()
#         #         max_in_vols = max(neigh_pressure)
#         #         min_in_vols = min(neigh_pressure)
#         #         if len(adj_volumes) < 4:
#         #             vol_verts = self.mtu.get_bridge_adjacencies(
#         #                 volume, 3, 0
#         #             )
#         #             adj_vols_verts = [
#         #                 self.mtu.get_bridge_adjacencies(
#         #                     adj_volume, 3, 0
#         #                     ) for adj_volume in adj_volumes
#         #             ]
#         #             adj_vols_verts.append(vol_verts)
#         #             adj_vols_verts = np.asarray(
#         #                 adj_vols_verts, dtype='uint64'
#         #             ).flatten()
#         #             adj_vols_verts_as_set = set(adj_vols_verts)
#         #             boundary_verts = adj_vols_verts_as_set.intersection(
#         #                 self.dirichlet_nodes
#         #             )
#         #             if boundary_verts:
#         #                 boundary_verts_as_list = list(boundary_verts)
#         #                 verts_pressure = np.asarray([
#         #                     self.get_boundary_node_pressure(node)
#         #                     for node in boundary_verts_as_list
#         #                 ]).flatten()
#         #                 max_in_verts = max(verts_pressure)
#         #                 min_in_verts = min(verts_pressure)
#         #         try:
#         #             self.u_maxs.update(
#         #                 {volume_id[0][0]: max(max_in_vols, max_in_verts)}
#         #             )
#         #             self.u_mins.update(
#         #                 {volume_id[0][0]: min(min_in_vols, min_in_verts)}
#         #             )
#         #         except UnboundLocalError:
#         #             self.u_maxs.update(
#         #                 {volume_id[0][0]: max_in_vols}
#         #             )
#         #             self.u_mins.update(
#         #                 {volume_id[0][0]: min_in_vols}
#         #             )
#         #     rows = []
#         #     cols = []
#         #     alpha_ijs = []
#         #     [
#         #         (
#         #             rows.append(i),
#         #             cols.append(j),
#         #             alpha_ijs.append(
#         #                 self.compute_alpha(i, j)
#         #             ),
#         #         ) for i, j in zip(_is, js) if i != j
#         #     ]
#         #     # print(f"max alpha: {max(alpha_ijs)} min alpha: {min(alpha_ijs)}")
#         #     # print("vai recalcular T_plus")
#         #     t0 = time.time()
#         #     # new_vals = [
#         #     #     self.T_plus[i, j] * alpha for i, j, alpha in zip(
#         #     #         rows, cols, alpha_ijs
#         #     #     )
#         #     # ]
#         #     # max(self.u[:len(self.volumes)])
#         #     # min(self.u[:len(self.volumes)])
#         #     # self.T_expanded[1579,1579]
#         #     import pdb;
#         #     self.T_plus_aux = self.T_plus.copy()
#         #     aux = self.T_plus[rows, cols].copy()
#         #     aux = np.multiply(alpha_ijs, self.T_plus[rows, cols].copy().toarray()).flatten()
#         #     self.T_plus_aux[rows, cols] = aux
#         #     self.T_plus_aux[range(len(self.u)), range(len(self.u))] = -self.T_plus_aux.sum(axis=1).transpose() + self.T_plus_aux.diagonal()
#         #     # print("Refaz o assembly da matry com novos fluxos")
#         #     pdb.set_trace()
#         #     self.T_expanded = self.T_minus + self.T_plus_aux
#         #     # print("calculo do termo fonte")
#         #     _q = (
#         #         -self.T_expanded[: len(self.volumes), len(self.volumes):]
#         #         * nodes_pressure
#         #     )
#         #     print("calculo de pressao u")
#         #     self.u[: len(self.volumes)] = spsolve(
#         #         self.T_expanded[: len(self.volumes), : len(self.volumes)], _q
#         #     )
#         #     residual = np.asarray(_q).flatten() - (self.T_expanded * self.u)[:len(self.volumes)]
#         #     tol = np.dot(vols, abs(residual[:len(self.volumes)]))
#         #     # print(f"res: {np.average(abs(residual[:len(self.volumes)]))}")
#         #     print(f"res: {tol}")
#         #     self.mb.tag_set_data(self.pressure_tag, self.volumes, self.u[:len(self.volumes)])
#         #     self.record_data(f'results_crr_{myCounter}.vtk')
#         #     myCounter += 1
#         #     if myCounter > 20:
#         #         import pdb; pdb.set_trace()
#         #         exit()
#         # print("DONE!!!")
#         # import pdb; pdb.set_trace()
#
#         # self.mb.tag_set_data(self.pressure_tag, self.volumes, self.x)
#
#         # self.mb.tag_set_data(self.pressure_tag, self.volumes, self.x)
#         # self.tag_velocity()
"""This is the begin."""
import pdb
import matplotlib.pyplot as plt
from pymoab import types
from PyTrilinos import Epetra, AztecOO, Amesos
import solvers.helpers.geometric as geo
import numpy as np
import time

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


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
        self.node_pressure_tag = mesh_data.node_pressure_tag

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
        self.aux_q = {vol: [] for vol in range(len(self.volumes))}
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
            self.aux_q[int(id_left)].append({node: RHS * pressure})
            self.aux_q[int(id_right)].append({node: -RHS * pressure})
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
            self.aux_q[int(id_left)].append({node: RHS * neu_term})
            self.aux_q[int(id_right)].append({node: -RHS * neu_term})
            # self.Q[id_right, 0] += - RHS * neu_term
            # self.Q[id_left, 0] += RHS * neu_term

            for volume, weight_N in self.nodes_ws[node].items():
                self.ids.append([id_left, id_right])
                v_id = self.mb.tag_get_data(self.global_id_tag, volume)[0][0]
                self.v_ids.append([v_id, v_id])
                self.ivalues.append([-RHS * weight_N, RHS * weight_N])

    def get_global_rows(self):
        aux_list = []
        epetra_global_rows = np.asarray(
            [
                self.T.ExtractGlobalRowCopy(row)[1]
                for row in range(len(self.volumes))]
        )
        for row, arrs in enumerate(epetra_global_rows):
            for val in arrs:
                aux_list.append([row, val])
        rows = np.asarray(aux_list)[:, 0]
        cols = np.asarray(aux_list)[:, 1]
        return rows, cols

    @staticmethod
    def create_sp_matrix(shape, dtype=np.float64):
        return lil_matrix((shape, shape), dtype=dtype)

    def copy_mat(self, rows, cols, shape=None, dtype=np.float64):
        A = self.create_sp_matrix(shape, dtype)
        A[rows, cols] = [self.T[row][col] for row, col in zip(rows, cols)]
        return A

    def solve_original_problem(self, mat: lil_matrix, q: np.array):

        if any([mat.shape[0] < len(self.volumes), len(q) < len(self.volumes)]):
            raise (
                "not solvable. matrix length must be at leat the size of the original problem"
            )

        x = spsolve(
            mat[:len(self.volumes), :len(self.volumes)].tocsc(), q[:len(self.volumes)]
        )
        return x

    def tag_verts_pressure(self):
        print("Will tag vertices pressure")
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
        print("Done tagging vertices pressure!!")

    def compute_mi(self, a_volume):
        vol_faces = self.mtu.get_bridge_adjacencies(a_volume, 2, 2)
        vol_nodes = self.mtu.get_bridge_adjacencies(a_volume, 0, 0)
        vol_crds = self.mb.get_coords(vol_nodes)
        vol_crds = np.reshape(vol_crds, ([4, 3]))
        vol_volume = self.mesh_data.get_tetra_volume(vol_crds)
        vol_centre = self.mb.get_coords(a_volume)
        mi = []
        for face in vol_faces:
            I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)
            L = list(set(vol_nodes).difference(
                set(self.mtu.get_bridge_adjacencies(face, 2, 0)))
            )
            JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
            JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
            LJ = self.mb.get_coords([J]) - self.mb.get_coords(L)
            N_IJK = np.cross(JI, JK) / 2.0
            test = np.dot(LJ, N_IJK)
            if test < 0.0:
                I, K = K, I
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords(
                    [J]
                )
                N_IJK = np.cross(JI, JK) / 2.0
            adj_vol = set(self.mtu.get_bridge_adjacencies(face, 2, 3))\
                .difference(set([a_volume, ]))
            if not adj_vol:
                continue
            adj_vol_centre = self.mb.get_coords(adj_vol)
            r = vol_centre - adj_vol_centre
            dot_product = np.dot(N_IJK, r)
            mi.append(abs(dot_product))
        mi = sum(mi) / (vol_volume)
        return mi

    def compute_all_mi(self):
        self.mis = {volume: self.compute_mi(volume) for volume in self.volumes}

    def get_local_min_max(self, volume):
        # volume_pressure = self.mb.tag_get_data(self.pressure_tag, volume).flatten()
        # adj_volumes = self.mtu.get_bridge_adjacencies(volume, 2, 3)
        # volume_vertices = self.mtu.get_bridge_adjacencies(volume, 0, 0)
        # adj_volumes_pressure = self.mb.tag_get_data(self.pressure_tag, adj_volumes)
        # adj_volumes_verts = [
        #     self.mtu.get_bridge_adjacencies(adj_vol, 0, 0)
        #     for adj_vol in adj_volumes
        # ]
        # all_volumes_verts = set(
        #     np.asarray(adj_volumes_verts).flatten()
        # ).union(set(np.asarray(volume_vertices)))
        # all_vertices_in_patch_pressure = self.mb.tag_get_data(
        #     self.mesh_data.node_pressure_tag,
        #     np.asarray(list(all_volumes_verts), dtype=np.uint64)
        # ).flatten()
        # surrounding_pressure_patch = all_vertices_in_patch_pressure\
        #                                  .flatten().tolist()\
        #                              + adj_volumes_pressure\
        #                                  .flatten().tolist()
        # p_max_local = max(
        #     0, volume_pressure + max(
        #         [adj_p - volume_pressure for adj_p in surrounding_pressure_patch]
        #     )
        # )
        # p_min_local = min(
        #     0, volume_pressure + min(
        #         [adj_p - volume_pressure for adj_p in surrounding_pressure_patch]
        #     )
        # )
        try:
            volumes_in_patch = [
                *self.mtu.get_bridge_adjacencies(volume, 0, 3), *volume
            ]
        except:
            volumes_in_patch = [
                *self.mtu.get_bridge_adjacencies(volume, 0, 3), volume
            ]
        vols_pressure = self.mb.tag_get_data(
            self.pressure_tag, volumes_in_patch
        )
        p_max_local = max(vols_pressure)
        p_min_local = min(vols_pressure)
        return p_max_local, p_min_local

    def compute_slip_fact(self, face):
        if face in self.dirichlet_faces | self.neumann_faces:
            volume = self.mtu.get_bridge_adjacencies(face, 2, 3)
            _u = self.mb.tag_get_data(self.pressure_tag, volume)
            max_v, min_v = self.get_local_min_max(volume)
            verts = self.mtu.get_bridge_adjacencies(face, 2, 0)
            avg_verts_pressure = self.mb.tag_get_data(
                self.mesh_data.node_pressure_tag, verts
            ).mean()
            lp = _u - avg_verts_pressure
            volume = np.asarray(volume)[0]
            if _u > avg_verts_pressure:
                # lfs = (max_v - _u)
                lfs = 2 * self.mis[volume] * (max_v - _u)
                _s = min(lfs, lp)
            else:
                # rts = (min_v - _u)
                rts = 2 * self.mis[volume] * (min_v - _u)
                _s = max(lp, rts)
        else:
            volume, adj = self.mtu.get_bridge_adjacencies(face, 2, 3)
            _u = self.mb.tag_get_data(self.pressure_tag, volume)
            max_v, min_v = self.get_local_min_max(volume)
            max_v_adj, min_v_adj = self.get_local_min_max(adj)
            u_adj = self.mb.tag_get_data(self.pressure_tag, adj)
            lp = _u - u_adj
            if _u > u_adj:
                lfs = 2 * self.mis[volume] * (max_v - _u)
                rts = 2 * self.mis[adj] * (u_adj - min_v_adj)
                # lfs = (max_v - _u)
                # rts = (u_adj - min_v_adj)
                _s = min(lfs, lp, rts)
            else:
                lfs = 2 * self.mis[volume] * (min_v - _u)
                rts = 2 * self.mis[adj] * (u_adj - max_v)
                # lfs = (min_v - _u)
                # rts = (u_adj - max_v)
                _s = max(lfs, lp, rts)
        slip_factor = (_s + 1e-20) / (lp + 1e-20)
        if abs(slip_factor) < 1e-15:
            slip_factor = 0
        return slip_factor

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
                # self.aux_q[volume_id].append({node: -RHS * neu_term})
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
            self.aux_q[id_volume].append({node: RHS for node in face_nodes})
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
            self.aux_q[id_volume].append({node: -RHS for node in [I, J, K]})
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

    def get_only_positive_off_diagonal_values(self, rows, cols):
        positive_values = np.asarray(
            [
                [row, col] for row, col
                in zip(rows, cols) if self.T[row, col] > 0
                and row != col
            ]
        )
        positive_rows = positive_values[:, 0]
        positive_cols = positive_values[:, 1]
        return positive_rows, positive_cols

    def sum_into_diagonal(self, mat: lil_matrix):
        size = mat.shape[0]
        diagonals = [-np.sum(mat[row].toarray()) for row in range(mat.shape[0])]
        mat[range(size), range(size)] = diagonals
        mat.tocsc()

    def defect_correction(self):
        self.compute_all_mi()
        rows, cols = self.get_global_rows()
        A = self.copy_mat(rows, cols, shape=len(self.volumes))
        q = np.asarray(self.Q)
        p_rows, p_cols = self.get_only_positive_off_diagonal_values(
            rows, cols
        )
        A_plus = self.copy_mat(p_rows, p_cols, shape=len(self.volumes))
        self.sum_into_diagonal(A_plus)
        A_minus = A - A_plus
        x_minus = self.solve_original_problem(A_minus, q)
        self.mb.tag_set_data(self.pressure_tag, self.volumes, x_minus)
        self.tag_verts_pressure()
        all_faces = self.dirichlet_faces.union(
            self.neumann_faces.union(self.intern_faces)
        )
        for face in all_faces:
            sl = self.compute_slip_fact(face)
            try:
                row, col = self.mb.tag_get_data(
                    self.global_id_tag, self.mtu.get_bridge_adjacencies(face, 2, 3)
                )
                A_plus[row, col] *= sl
            except ValueError:
                # self.aux_q[id_volume].append({node: -RHS for node in [I, J, K]})
                pass
    def savefig(self, df, column_name, figname):

        hist = df[column_name].hist(bins=15)
        fig = hist.get_figure()
        fig.savefig(figname)
        plt.close()



# sol = []
# for face in all_faces:
#     sl = self.compute_slip_fact(face)
#     print(sl)
#     try:
#         sol.append([face, sl[0][0]])
#     except TypeError:
#         sol.append([face, sl])
# print('done')
# df = pd.DataFrame(sol, columns=['face_id', 'alpha'])
# self.savefig(df, 'alpha', 'distribuicao_alpha_sol_A.png')
