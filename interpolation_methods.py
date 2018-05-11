import numpy as np
from itertools import cycle
from pymoab import types
from perssure_solver_3D import MpfaD3D


class InterpolMethod(MpfaD3D):

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

    def eta(self, volume, opposite_vertice, node, j_aux):
        pass

    def xi(self, volume, opposite_vertice, r_aux):
        pass

    def _get_opposite_area_vector(self, opposite_vert):
        pass

    def by_lpew2(self, node):
        if node in self.dirichlet_nodes:
            print('a dirichlet node')
        if node in self.neumann_nodes:
            print('a neumann node')
        elif node in self.intern_nodes:
            vols_around = self.mtu.get_bridge_adjacencies(node, 0, 3)
            weights = np.array([])
            weight_sum = 0.0
            T = {}
            for a_vol in vols_around:
                k_hat = a_vol
                k_hat_centroid = self.mesh_data.get_centroid(k_hat)
                adj_vols = self._get_volumes_sharing_face_and_node(node, a_vol)
                adj_vols = [list(adj_vols[i])[0] for i in range(len(adj_vols))]
                aux_verts = list(set(self.mtu.get_bridge_adjacencies(a_vol,
                                     3, 0)).difference(set([node])))
                _aux_verts = cycle(aux_verts)
                for i, aux_vert in zip(range(3), _aux_verts):
                    aux1 = list(T.keys())[list(T.values()).index(aux_vert)]
                    aux2 = list(T.keys())[list(T.values()).index(
                                next(_aux_verts))]
                    if aux1 + aux2 != 6:
                        index = (aux1 + aux2) / 2
                        T[index] = self.mtu.get_average_position([node,
                                                                  aux1, aux2])
                        # might be more interesting getitng the real aentity
                        # (face, edge, node,...)
                    else:
                        T[6] = self.mtu.get_average_position([node,
                                                             aux1, aux2])
                for index, aux in zip([1, 3, 5], aux_verts):
                    T[index] = self.mtu.get_average_position([aux, node])



    def _lambda_lpew3(self, node, aux_node, face):
        adj_vols = self.mtu.get_bridge_adjacencies(face, 2, 3)
        adj_nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
        ref_node = set(adj_vols) - set([node, aux_node])
        for a_vol in adj_vols:
            vol_cent = self.mesh_data.get_centroid(a_vol)
            vol_nodes = self.mesh_data.mb.get_adjacencies(a_vol, 0)
            vol_nodes_crds = self.mesh_data.mb.get_coords(vol_nodes)
            vol_nodes_crds = np.reshape(vol_nodes_crds, (4, 3))
            tetra_vol = self.mesh_data.mb._adjacencies(vol_nodes_crds)

    pass




    def by_lpew3(self, node):
        adj_vols = self.mtu.get_bridge_adjacencies(node, 0, 3)
        for a_vol in adj_vols:
            aux_verts = set(self.mb.get_adjacencies(a_vol, 0))
            aux_verts.remove(node)

        pass
