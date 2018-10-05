import numpy as np
import mpfad.helpers.geometric as geo
from itertools import cycle
# from pymoab import types
from .InterpolationMethod import InterpolationMethodBase


class LPEW2(InterpolationMethodBase):

    def _getVolumesSharingFaceAndNode(self, node, volume):
        vols_around_node = self.mtu.get_bridge_adjacencies(node, 0, 3)
        adj_vols = self.mtu.get_bridge_adjacencies(volume, 2, 3)
        volumes_sharing_face_and_node = set(adj_vols).difference(
                                        set(adj_vols).difference(
                                            set(vols_around_node)))
        return list(volumes_sharing_face_and_node)

    def _getAuxiliaryVerts(self, node, volume):
        # create meshset for volume and node
        T = {}
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
            aux1 = list(T.keys())[list(T.values()).index(aux_verts[0])]
            aux2 = list(T.keys())[list(T.values()).index(aux_verts[1])]
            if aux1 + aux2 != 6:
                index = (aux1 + aux2) / 2
                T[int(index)] = aux_face
            else:
                T[6] = aux_face
        T[7] = T[1]

        self.T = T

    def _getAuxiliaryVertsCoords(self, node, volume, tao):
        coords = {}
        for index in [1, 3, 5, 7]:
            coords[index] = self.mtu.get_average_position([node, self.T[index]]
                                                          )
        for index in [2, 4, 6]:
            coords[index] = self.mtu.get_average_position([self.T[index]])
        return coords

    def _getAuxiliaryVolumes(self, T, node):
        aux_vols = {}
        aux_vols[0] = [T[2], T[4], T[6], node]
        for i in range(1, 4):
            try:
                aux_vols[i] = [T[2 * i], T[2 * i - 1], T[2 * i - 2], node]
            except KeyError:
                aux_vols[i] = [T[2], T[1], T[6], node]
        aux_vols[4] = aux_vols[1]
        return aux_vols

    def _computeA(self, j, vols, perm):
        if j % 2 == 0:
            A = (self._csi_lpew2(vols[0], self.T[j], perm) +
                 self._csi_lpew2(vols[int(j / 2)], self.T[j], perm)
                 + self._csi_lpew2(vols[int(j / 2 + 1)], self.T[j], perm))
        else:
            A = (self._csi_lpew2(vols[int((j + 1) / 2)], self.T[j], perm))
        return A

    def _fluxTerm(self, vector_1st, permeab, vector_2nd, vol):
        aux_1 = np.dot(vector_1st, permeab)
        aux_2 = np.dot(aux_1, vector_2nd)
        fluxTerm = aux_2 / vol
        return fluxTerm

    def _rRange(sefl, j, r):
        size = ((j - r + 5) % 6) + 1
        last_val = ((j + 4) % 6) + 1

        prod = [((last_val - k) % 6) + 1 for k in range(1, size+1)]
        return prod[::-1]

    def _phi(self, r, vol, adjVols, node):
        TrCoords = self.T_coords[r]
        TrPlusOneCoords = self.T_coords[r + 1]
        nodeCoords = self.mb.get_coords([node])
        rFace = np.asarray([TrCoods, TrPlusOneCoords, nodeCoords]).reshape([3,3])
        # compute neta_vol_r
        # compute neta_vol_r_plus_1
        # compute neta_adj_vol_r
        # compute neta_adj_vol_r_plus_1
        # compute phi
        pass

    def _prodPhi(self, r_star, j, vol_perm, adj_vol_perm,
                  vol_centre, adj_vol_centre):
        # call rRange
        # phiProd = 1.
        # for rStar in rRange:
        #     phi = self._phi_lpew2(self, rStar, vol_perm,
        #                           adj_vol_perm, vol_centre, adj_vol_centre)
        # phiProd = phiProd * phi
        pass

    def _neta_lpew2(self, r, perm, volume_centre, node):
        # construct face based on r. r is the auxiliary counter.
        # The face must be formed by r, r + 1 and node auxiliary verts.
        # sub_face_verts = np.asarray([T[r], T[r + 1], node])
        # compute subvolume volume. coords are: sub_face_verts + volume_centre
        # compute normal face vector opposite to volume centre
        # compute normal opposite to auxiliary Tr
        # compute neta

        pass

    def _csi_lpew2(self, vol_verts, aux_vert, perm):
        vol_nodes_coords = np.asarray([self.mb.get_coords([vol])
                                       for vol in vol_verts])
        aux_vert_coords = self.mb.get_coords([aux_vert])
        vol_nodes_coords.reshape([4, 3])
        tetra_volume = geo.get_tetra_volume(vol_nodes_coords)
        N_node = geo._area_vector(vol_nodes_coords, vol_nodes_coords[3])[0]
        verts_opposite_to_aux_vert = list(set(vol_verts).difference({aux_vert})
                                          )
        opposite_face_coords = np.asarray([self.mb.get_coords([vol])
                                           for vol in
                                           verts_opposite_to_aux_vert])
        N_Tj = geo._area_vector(opposite_face_coords, aux_vert_coords)[0]
        csi = self._fluxTerm(N_node, perm, N_Tj, tetra_volume)
        return csi

    def interpolate(self, node, tao=0.5):
        volsAroundNode = self.mtu.get_bridge_adjacencies(node, 0, 3)
        partialWts = np.zeros([len(volsAroundNode)])
        mapIds = {volAroundNode: _id for volAroundNode, _id in
                  zip(volsAroundNode, range(len(volsAroundNode)))}
        A = np.zeros([len(volsAroundNode), 6])
        for count, volAroundNode in zip(range(len(volsAroundNode)),
                                        volsAroundNode):
            perm = np.asarray(self.mb.tag_get_data(self.perm_tag,
                                                   volAroundNode))
            perm = perm.reshape([3, 3])
            self._getAuxiliaryVerts(node, volAroundNode)
            self._getAuxiliaryVertsCoords(node, volAroundNode, tao)
            aux_vols = self._getAuxiliaryVolumes(self.T, node)
            for j in range(1, 7):
                A = self._computeA(j, aux_vols, perm)
                adjVols = self._getVolumesSharingFaceAndNode(node,
                                                             volAroundNode)
