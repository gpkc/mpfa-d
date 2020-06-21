import numpy as np
import solvers.helpers.geometric as geo
from .InterpolationMethod import InterpolationMethodBase
from functools import lru_cache

# import celery and add configs


class LPEW3(InterpolationMethodBase):
    def flux_term(self, vector_1st, permeab, vector_2nd, face_area=1.0):
        aux_1 = np.dot(vector_1st, permeab)
        aux_2 = np.dot(aux_1, vector_2nd)
        flux_term = aux_2 / face_area
        return flux_term

    @lru_cache(maxsize=200)
    def lambda_lpew3(self, node, aux_node, face):
        # Should include tao parameter in order to change interpolation region
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
            vol_cent = self.mesh_data.mb.tag_get_data(
                self.mesh_data.volume_centre_tag, a_vol
            )[0]
            vol_nodes = self.mb.get_adjacencies(a_vol, 0)
            sub_vol = np.append(face_nodes_crds, vol_cent)
            sub_vol = np.reshape(sub_vol, (4, 3))
            tetra_vol = self.mesh_data.get_tetra_volume(sub_vol)
            ref_node_i = list(set(vol_nodes) - set(face_nodes))
            ref_node_i = self.mb.get_coords(ref_node_i)
            N_int = geo._area_vector([node, aux_node, vol_cent], ref_node)[0]
            N_i = geo._area_vector(face_nodes_crds, ref_node_i)[0]
            lambda_l += self.flux_term(N_i, vol_perm, N_int) / tetra_vol
        return lambda_l

    @lru_cache(maxsize=200)
    def neta_lpew3(self, node, vol, face):
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
        N_out = geo._area_vector(face_nodes_i, node)[0]
        N_i = geo._area_vector(face_nodes_crds, ref_node)[0]
        neta = self.flux_term(N_out, vol_perm, N_i) / tetra_vol
        return neta

    @lru_cache(maxsize=200)
    def csi_lpew3(self, face, vol):
        vol_perm = self.mb.tag_get_data(self.perm_tag, vol)
        vol_perm = np.reshape(vol_perm, (3, 3))
        vol_cent = self.mesh_data.mb.tag_get_data(
            self.mesh_data.volume_centre_tag, vol
        )[0]
        face_nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
        face_nodes = self.mb.get_coords(face_nodes)
        face_nodes = np.reshape(face_nodes, (3, 3))
        N_i = geo._area_vector(face_nodes, vol_cent)[0]
        sub_vol = np.append(face_nodes, vol_cent)
        sub_vol = np.reshape(sub_vol, (4, 3))
        tetra_vol = self.mesh_data.get_tetra_volume(sub_vol)
        csi = self.flux_term(N_i, vol_perm, N_i) / tetra_vol
        return csi

    @lru_cache(maxsize=200)
    def sigma_lpew3(self, node, vol):
        node_crds = self.mb.get_coords([node])
        adj_faces = set(self.mtu.get_bridge_adjacencies(node, 0, 2))
        vol_faces = set(self.mtu.get_bridge_adjacencies(vol, 3, 2))
        in_faces = list(adj_faces & vol_faces)
        vol_cent = self.mesh_data.mb.tag_get_data(
            self.mesh_data.volume_centre_tag, vol
        )[0]
        clockwise = 1.0
        counter_clockwise = 1.0
        for a_face in in_faces:
            aux_nodes = set(self.mtu.get_bridge_adjacencies(a_face, 2, 0))
            aux_nodes.remove(node)
            aux_nodes = list(aux_nodes)
            aux_nodes_crds = self.mb.get_coords(aux_nodes)
            aux_nodes_crds = np.reshape(aux_nodes_crds, (2, 3))
            aux_vect = [node_crds, aux_nodes_crds[0], aux_nodes_crds[1]]
            spin = geo._area_vector(aux_vect, vol_cent)[1]
            if spin < 0:
                aux_nodes[0], aux_nodes[1] = aux_nodes[1], aux_nodes[0]
            count = self.lambda_lpew3(node, aux_nodes[0], a_face)
            counter_clockwise = counter_clockwise * count
            clock = self.lambda_lpew3(node, aux_nodes[1], a_face)
            clockwise = clockwise * clock
        sigma = clockwise + counter_clockwise
        return sigma

    @lru_cache(maxsize=200)
    def phi_lpew3(self, node, vol, face):
        if len(vol) == 0:
            return 0.0
        vol = vol[0]
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
            lbd = self.lambda_lpew3(node, aux_node[0], a_face)
            lambda_mult = lambda_mult * lbd
        sigma = self.sigma_lpew3(node, vol)
        neta = self.neta_lpew3(node, vol, face)
        phi = lambda_mult * neta / sigma
        return phi

    @lru_cache(maxsize=200)
    def psi_sum_lpew3(self, node, vol, face):
        if len(vol) == 0:
            return 0.0
        vol = vol[0]
        face_nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
        vol_nodes = self.mb.get_adjacencies(vol, 0)
        aux_node = set(vol_nodes) - set(face_nodes)
        aux_node = list(aux_node)

        adj_faces = set(self.mtu.get_bridge_adjacencies(node, 0, 2))
        vol_faces = set(self.mtu.get_bridge_adjacencies(vol, 3, 2))
        in_faces = adj_faces & vol_faces
        faces = in_faces - set([face])
        faces = list(faces)
        psi_sum = 0.0
        for i in range(len(faces)):
            a_face_nodes = self.mtu.get_bridge_adjacencies(faces[i], 2, 0)
            other_node = set(face_nodes) - set(a_face_nodes)
            other_node = list(other_node)
            lbd_1 = self.lambda_lpew3(node, aux_node[0], faces[i])
            lbd_2 = self.lambda_lpew3(node, other_node[0], faces[i - 1])
            neta = self.neta_lpew3(node, vol, faces[i])
            psi = lbd_1 * lbd_2 * neta
            psi_sum += +psi
        sigma = self.sigma_lpew3(node, vol)
        psi_sum = psi_sum / sigma
        return psi_sum

    def partial_weight_lpew3(self, node, vol):
        vol_faces = self.mtu.get_bridge_adjacencies(vol, 3, 2)
        adj_faces = self.mtu.get_bridge_adjacencies(node, 0, 2)
        vol_node_faces = set(vol_faces) & set(adj_faces)
        zepta = 0.0
        delta = 0.0
        for a_face in vol_node_faces:
            a_neighs = self.mtu.get_bridge_adjacencies(a_face, 2, 3)
            a_neighs = np.asarray(a_neighs, dtype="uint64")
            a_neigh = a_neighs[a_neighs != vol]
            csi = self.csi_lpew3(a_face, vol)
            psi_sum_neigh = self.psi_sum_lpew3(node, tuple(a_neigh), a_face)
            psi_sum_vol = self.psi_sum_lpew3(node, (vol,), a_face)
            zepta += (psi_sum_vol + psi_sum_neigh) * csi
            phi_vol = self.phi_lpew3(node, (vol,), a_face)
            phi_neigh = self.phi_lpew3(node, tuple(a_neigh), a_face)
            delta += (phi_vol + phi_neigh) * csi
        p_weight = zepta - delta
        return p_weight

    def neumann_treatment(self, node):
        adj_faces = self.mtu.get_bridge_adjacencies(node, 0, 2)
        N_term_sum = 0.0
        for face in adj_faces:
            if face not in self.neumann_faces:
                continue
            face_flux = self.mb.tag_get_data(self.neumann_tag, face)[0][0]
            face_nodes = self.mb.get_adjacencies(face, 0)
            nodes_crds = self.mb.get_coords(face_nodes)
            nodes_crds = np.reshape(nodes_crds, (len(face_nodes), 3))
            face_area = geo._area_vector(
                nodes_crds, np.array([0.0, 0.0, 0.0]), norma=True
            )
            # store face_area
            vol_N = tuple(self.mtu.get_bridge_adjacencies(face, 2, 3))
            psi_N = self.psi_sum_lpew3(node, vol_N, face)
            phi_N = self.phi_lpew3(node, vol_N, face)
            N_term = -3.0 * (1 + (psi_N - phi_N)) * face_flux * face_area
            N_term_sum += N_term
        return N_term_sum

    def interpolate(self, node, neumann=False):
        # vols_around_ms = self.mb.create_meshset()
        vols_around = self.mtu.get_bridge_adjacencies(node, 0, 3)
        # self.mb.add_entities(vols_around_ms, vols_around)
        weights = [
            self.partial_weight_lpew3(node, a_vol) for a_vol in vols_around
        ]
        # weights = [
        #     max(0.0, self.partial_weight_lpew3(node, a_vol))
        #     for a_vol in vols_around
        # ]  # for undershooting
        weight_sum = np.sum(weights)
        weights = weights / weight_sum
        # for overshooting
        # self.mb.tag_set_data(self.node_wts_tag, vols_around, weights)
        node_weights = {
            vol: weight for vol, weight in zip(vols_around, weights)
        }
        if neumann:
            neu_term = self.neumann_treatment(node) / weight_sum
            node_weights[node] = neu_term
        # print(node_weights)
        return node_weights
