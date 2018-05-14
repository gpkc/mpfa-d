import numpy as np
from itertools import cycle
from pymoab import types
from pymoab import rng
from pressure_solver_3D import MpfaD3D


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

    def _neta_lpew2(self, node, vol, face, reff_node):
        vol_perm = self.mb.tag_get_data(self.perm_tag, vol)
        vol_perm = np.reshape(vol_perm, (3, 3))
        vol_centroid = self.mesh_data.get_centroid(vol)
        face_nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
        face_node_coords = self.mb.get_coords(face_nodes)
        face_node_coords.reshape([3,3])
        vol_coords = np.append(face_node_coords, vol_centroid)
        tetra_volume = self.mesh_data.get_tetra_volume(vol_coords.reshape([4,3]))
        opposite_face_nodes = set(face_nodes).difference(set([reff_node]))
        opposite_face_nodes_coords = self.mb.get_coords(opposite_face_nodes)
        opposite_face_nodes_coords = np.append(opposite_face_nodes_coords,
                                               vol_centroid).reshape([3,3])
        area_vector_reff_node = self._area_vector(opposite_face_nodes_coords,
                                                  self.mb.get_coords([reff_node
                                                                      ]))[0]
        area_vector_centroid = self._area_vector(face_node_coords.reshape(
                                                 [3, 3]), vol_centroid)[0]
        neta = self._flux_term(area_vector_reff_node,
                               vol_perm, area_vector_centroid
                               )/(3.0 * tetra_volume)
        return neta


    def xi_lpew2(self, volume, opposite_vertice, face):
        pass
        # return xi

    def _phi_lpew2(self, neta_k_r, neta_k_r_plus_1,
                   neta_k_j_r, neta_k_j_r_plus_1):
        if self.neumann:
            phi = neta_k_r / neta_k_r_plus_1
        else:
            phi = (neta_k_j_r - neta_k_r) / (neta_k_j_r_plus_1 - neta_k_r_plus_1)

        return phi

    def _get_opposite_area_vector(self, opposite_vert):
        pass

    def _arrange_auxiliary_faces_and_verts(self, node, volume):
        T = {}
        adj_vols = self._get_volumes_sharing_face_and_node(node, volume)
        adj_vols = [list(adj_vols[i])[0] for i in range(len(adj_vols))]
        aux_verts_bkp = list(set(self.mtu.get_bridge_adjacencies(volume,
                                 3, 0)).difference(set([node])))
        aux_faces = list(set(self.mtu.get_bridge_adjacencies(volume,
                                 3, 2)))
        for face in aux_faces:
            nodes_in_aux_face = list(set(self.mtu.get_bridge_adjacencies(
                                         face, 2, 0)).difference(set([node])
                                                                 )
                                         )
            if len(nodes_in_aux_face) == 3:
                    aux_faces.remove(face)
        aux_verts = cycle(aux_verts_bkp)
        for index, aux_vert in zip([1, 3, 5], aux_verts):
            T[index] = aux_vert
        for aux_face in aux_faces:
            aux_verts = list(set(self.mtu.get_bridge_adjacencies(
                                         aux_face, 2, 0)).difference(set(
                                                                     [node]
                                                                     )))
            adj_vol = set(self.mtu.get_bridge_adjacencies(aux_face, 2, 3)).difference(set([volume]))
            aux1 = list(T.keys())[list(T.values()).index(aux_verts[0])]
            aux2 = list(T.keys())[list(T.values()).index(aux_verts[1])]
            if aux1 + aux2 != 6:
                index = (aux1 + aux2) / 2
                T[int(index)] = (aux_face,
                                 self.mtu.get_average_position([aux_face]), adj_vol)
            else:
                T[6] = (aux_face, self.mtu.get_average_position([aux_face]), adj_vol)
        aux_verts = cycle(aux_verts_bkp)
        for index, aux_vert in zip([1, 3, 5], aux_verts):
            T[index] = (aux_vert,
                        self.mtu.get_average_position([node, aux_vert])
                        )
        vol_centroid = np.asarray(self.mesh_data.get_centroid(volume))
        print(vol_centroid)
        coords = [T[1][1], T[2][1], T[3][1], T[4][1], T[5][1], T[6][1]]
        verts = self.mb.create_vertices(coords)
        faces = [self.mb.create_element(types.MBTRI,
                 [node, verts[0], verts[1]]),
                 self.mb.create_element(types.MBTRI,
                  [node, verts[1], verts[2]]),
                 self.mb.create_element(types.MBTRI,
                  [node, verts[2], verts[3]]),
                 self.mb.create_element(types.MBTRI,
                  [node, verts[3], verts[4]]),
                 self.mb.create_element(types.MBTRI,
                  [node, verts[4], verts[5]]),
                 self.mb.create_element(types.MBTRI,
                  [node, verts[5], verts[0]])
                  ]
        t = {}
        for i in range(1, 7, 1):
            try:
                t[i] = (verts[i - 1], faces[i-1], T[i][2])
            except:
                t[i] = (verts[i - 1], faces[i-1])

        """
        self.mtu.construct_aentities(rng.Range([ed1, ed2, ed3, ed4, ed5, ed6]))
        self.mb.write_file("weird.vtk")
        print(verts)
        """
        return t

    def A_aux(self):
        return 1.0

    def by_lpew2(self, node):
        if node in self.dirichlet_nodes:
            print('a dirichlet node')
        if node in self.neumann_nodes:
            self.neumann = True
            print('a neumann node')
        elif node in self.intern_nodes:
            self.neumann = False
            vols_around = self.mtu.get_bridge_adjacencies(node, 0, 3)
            weights = np.array([])
            weight_sum = 0.0
            for a_vol in vols_around:
                T = self._arrange_auxiliary_faces_and_verts(
                                      node, a_vol)
                cycle_r = cycle([1, 2, 3, 4, 5, 6])
                cycle_r_plus_1 = cycle([2, 3, 4, 5, 6, 1])
                # for j_aux in range(1,7,1):
                    #for _, r, r_plus_1 in zip(range(6), cycle_r, cycle_r_plus_1):
                        # print(r, self.mb.get_coords([T[r][0]]), self._neta_lpew2(node,  a_vol, T[r][1], T[r][0]))


    def _area_vector(self, nodes, ref_node):
        ref_vect = nodes[0] - ref_node
        AB = nodes[1] - nodes[0]
        AC = nodes[2] - nodes[0]
        area_vector = np.cross(AB, AC) / 2.0
        if np.dot(area_vector, ref_vect) < 0.0:
            area_vector = - area_vector
            return [area_vector, -1]
        return [area_vector, 1]

    def _lambda_lpew3(self, node, aux_node, face):
        adj_vols = self.mtu.get_bridge_adjacencies(face, 2, 3)
        face_nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
        ref_node = list(set(face_nodes) - (set([node]) | set([aux_node])))
        face_nodes_crds = self.mb.get_coords(face_nodes)
        face_nodes_crds = np.reshape(face_nodes_crds, (3, 3))
        ref_node = self.mb.get_coords(ref_node)
        aux_node = self.mb.get_coords([aux_node])
        node = self.mb.get_coords([node])
        lambda_l = 0.0
        for a_vol in adj_vols:
            vol_perm = self.mb.tag_get_data(self.perm_tag, a_vol)
            vol_perm = np.reshape(vol_perm, (3, 3))
            vol_cent = self.mesh_data.get_centroid(a_vol)
            vol_nodes = self.mb.get_adjacencies(a_vol, 0)
            sub_vol = np.append(face_nodes_crds, vol_cent)
            sub_vol = np.reshape(sub_vol, (4, 3))
            tetra_vol = self.mesh_data.get_tetra_volume(sub_vol)
            ref_node_i = list(set(vol_nodes) - set(face_nodes))
            ref_node_i = self.mb.get_coords(ref_node_i)
            N_int = self._area_vector([node, aux_node, vol_cent], ref_node)[0]
            N_i = self._area_vector(face_nodes_crds, ref_node_i)[0]
            lambda_l += self._flux_term(N_i, vol_perm, N_int)/(3.0*tetra_vol)
        return lambda_l

    def _neta_lpew3(self, node, vol, face):
        vol_perm = self.mb.tag_get_data(self.perm_tag, vol)
        vol_perm = np.reshape(vol_perm, (3, 3))
        vol_nodes = self.mb.get_adjacencies(vol, 0)
        face_nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
        face_nodes_crds = self.mb.get_coords(face_nodes)
        face_nodes_crds = np.reshape(face_nodes_crds, (3, 3))
        ref_node = list(set(vol_nodes) - set(face_nodes))
        ref_node = self.mb.get_coords(ref_node)
        vol_nodes_crds = self.mb.get_coords(list(vol_nodes))
        vol_nodes_crds = np.reshape(vol_nodes_crds, (4, 3))
        tetra_vol = self.mesh_data.get_tetra_volume(vol_nodes_crds)
        vol_nodes = set(vol_nodes)
        vol_nodes.remove(node)
        face_nodes_i = self.mb.get_coords(list(vol_nodes))
        face_nodes_i = np.reshape(face_nodes_i, (3, 3))
        node = self.mb.get_coords([node])
        N_out = self._area_vector(face_nodes_i, node)[0]
        N_i = self._area_vector(face_nodes_crds, ref_node)[0]
        neta = self._flux_term(N_out, vol_perm, N_i)/(3.0 * tetra_vol)
        return neta

    def _csi_lpew3(self, face, vol):
        vol_perm = self.mb.tag_get_data(self.perm_tag, vol)
        vol_perm = np.reshape(vol_perm, (3, 3))
        vol_cent = self.mesh_data.get_centroid(vol)
        face_nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
        face_nodes = self.mb.get_coords(face_nodes)
        face_nodes = np.reshape(face_nodes, (3, 3))
        N_i = self._area_vector(face_nodes, vol_cent)[0]
        sub_vol = np.append(face_nodes, vol_cent)
        sub_vol = np.reshape(sub_vol, (4, 3))
        tetra_vol = self.mesh_data.get_tetra_volume(sub_vol)
        csi = self._flux_term(N_i, vol_perm, N_i)/(3.0*tetra_vol)
        return csi

    def _sigma_lpew3(self, node, vol):
        node_crds = self.mb.get_coords([node])
        adj_faces = set(self.mtu.get_bridge_adjacencies(node, 0, 2))
        vol_faces = set(self.mtu.get_bridge_adjacencies(vol, 3, 2))
        in_faces = list(adj_faces & vol_faces)
        vol_cent = self.mesh_data.get_centroid(vol)
        clockwise = 1.0
        counterwise = 1.0
        for a_face in in_faces:
            aux_nodes = set(self.mtu.get_bridge_adjacencies(a_face, 2, 0))
            aux_nodes.remove(node)
            aux_nodes = list(aux_nodes)
            aux_nodes_crds = self.mb.get_coords(aux_nodes)
            aux_nodes_crds = np.reshape(aux_nodes_crds, (2, 3))
            aux_vect = [node_crds, aux_nodes_crds[0], aux_nodes_crds[1]]
            clock_test = self._area_vector(aux_vect, vol_cent)[1]
            if clock_test < 0:
                aux_nodes[0], aux_nodes[1] = aux_nodes[1], aux_nodes[0]
            count = self._lambda_lpew3(node, aux_nodes[0], a_face)
            counterwise = counterwise * count
            clock = self._lambda_lpew3(node, aux_nodes[1], a_face)
            clockwise = clockwise * clock
        sigma = clockwise + counterwise
        return sigma

    def _phi_lpew3(self, node, vol, face):
        face_nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
        vol_nodes = self.mb.get_adjacencies(vol, 0)
        aux_node = set(vol_nodes) - set(face_nodes)
        aux_node = list(aux_node)
        adj_faces = set(self.mtu.get_bridge_adjacencies(node, 0, 2))
        vol_faces = set(self.mtu.get_bridge_adjacencies(vol, 3, 2))
        in_faces = adj_faces & vol_faces
        faces = in_faces - set([face])
        faces = list(faces)
        lambda_mult = 1.0
        for a_face in faces:
            lbd = self._lambda_lpew3(node, aux_node[0], a_face)
            lambda_mult = lambda_mult * lbd
        sigma = self._sigma_lpew3(node, vol)
        neta = self._neta_lpew3(node, vol, face)
        phi = lambda_mult * neta / sigma
        return phi

    def _psi_sum_lpew3(self, node, vol, face):
        face_nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
        vol_nodes = self.mb.get_adjacencies(vol, 0)
        aux_node = set(vol_nodes) - set(face_nodes)
        aux_node = list(aux_node)

        adj_faces = set(self.mtu.get_bridge_adjacencies(node, 0, 2))
        vol_faces = set(self.mtu.get_bridge_adjacencies(vol, 3, 2))
        in_faces = adj_faces & vol_faces
        faces = in_faces - set([face])
        faces = list(faces)
        phi_sum = 0.0
        for i in range(len(faces)):
            a_face_nodes = self.mtu.get_bridge_adjacencies(faces[i], 2, 0)
            other_node = set(face_nodes) - set(a_face_nodes)
            other_node = list(other_node)
            lbd_1 = self._lambda_lpew3(node, aux_node[0], faces[i])
            lbd_2 = self._lambda_lpew3(node, other_node[0], faces[i-1])
            neta = self._neta_lpew3(node, vol, faces[i])
            phi = lbd_1 * lbd_2 * neta
            phi_sum += + phi
        sigma = self._sigma_lpew3(node, vol)
        phi_sum = phi_sum / sigma
        return phi_sum

    def _partial_weight_lpew3(self, node, vol):
        vol_faces = self.mtu.get_bridge_adjacencies(vol, 3, 2)
        vols_neighs = self.mtu.get_bridge_adjacencies(vol, 2, 3)
        zepta = 0.0
        delta = 0.0
        for a_neigh in vols_neighs:
            neigh_faces = self.mtu.get_bridge_adjacencies(a_neigh, 3, 2)
            a_face = set(vol_faces) & set(neigh_faces)
            a_face = list(a_face)
            csi = self._csi_lpew3(a_face[0], vol)
            psi_sum_neigh = self._psi_sum_lpew3(node, a_neigh, a_face[0])
            psi_sum_vol = self._psi_sum_lpew3(node, vol, a_face[0])
            zepta += (psi_sum_vol + psi_sum_neigh) * csi
            phi_vol = self._phi_lpew3(node, vol, a_face[0])
            phi_neigh = self._phi_lpew3(node, a_neigh, a_face[0])
            delta += (phi_vol + phi_neigh) * csi
        p_weight = zepta - delta
        return p_weight

    def by_lpew3(self, node):
        vols_around = self.mtu.get_bridge_adjacencies(node, 0, 3)
        weights = np.array([])
        weight_sum = 0.0
        for a_vol in vols_around:
            p_weight = self._partial_weight_lpew3(node, a_vol)
            weights = np.append(weights, p_weight)
            weight_sum += p_weight
        weights = weights / weight_sum
        node_weights = {
            vol: weight for vol, weight in zip(vols_around, weights)}
        return node_weights
