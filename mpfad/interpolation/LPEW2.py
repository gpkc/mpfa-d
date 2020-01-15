import numpy as np
import mpfad.helpers.geometric as geo
from itertools import cycle

# from pymoab import types
from .InterpolationMethod import InterpolationMethodBase


class LPEW2(InterpolationMethodBase):
    def _getVolumesSharingFaceAndNode(self, node, volume):
        vols_around_node = self.mtu.get_bridge_adjacencies(node, 0, 3)
        adj_vols = self.mtu.get_bridge_adjacencies(volume, 2, 3)
        volumesSharingFaceAndNode = list(
            set(adj_vols).difference(
                set(adj_vols).difference(set(vols_around_node))
            )
        )
        mapAuxVols = {
            i: auxVol
            for i, auxVol in zip(range(1, 4), volumesSharingFaceAndNode)
        }
        return mapAuxVols

    def _getAuxiliaryVerts(self, node, volume):
        # create meshset for volume and node
        T = {}
        aux_verts_bkp = list(
            set(self.mtu.get_bridge_adjacencies(volume, 3, 0)).difference(
                set([node])
            )
        )
        aux_faces = list(set(self.mtu.get_bridge_adjacencies(volume, 3, 2)))
        for face in aux_faces:
            nodes_in_aux_face = list(
                set(self.mtu.get_bridge_adjacencies(face, 2, 0)).difference(
                    set([node])
                )
            )
            if len(nodes_in_aux_face) == 3:
                aux_faces.remove(face)
        aux_verts = cycle(aux_verts_bkp)
        for index, aux_vert in zip([1, 3, 5], aux_verts):
            T[index] = aux_vert
        for aux_face in aux_faces:
            aux_verts = list(
                set(
                    self.mtu.get_bridge_adjacencies(aux_face, 2, 0)
                ).difference(set([node]))
            )
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
            coords[index] = self.mtu.get_average_position(
                [node, self.T[index]]
            )
        for index in [2, 4, 6]:
            coords[index] = self.mtu.get_average_position([self.T[index]])

        self.T_coords = coords

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
            A = (
                self._csi_lpew2(vols[0], self.T[j], perm)
                + self._csi_lpew2(vols[int(j / 2)], self.T[j], perm)
                + self._csi_lpew2(vols[int(j / 2 + 1)], self.T[j], perm)
            )
        else:
            A = self._csi_lpew2(vols[int((j + 1) / 2)], self.T[j], perm)
        return A

    def _fluxTerm(self, vector_1st, permeab, vector_2nd, vol):
        aux_1 = np.dot(vector_1st, permeab)
        aux_2 = np.dot(aux_1, vector_2nd)
        fluxTerm = aux_2 / vol
        return fluxTerm

    def _rRange(sefl, r, j):
        size = ((j - r + 5) % 6) + 1
        last_val = ((j + 4) % 6) + 1

        prod = [((last_val - k) % 6) + 1 for k in range(1, size + 1)]
        return prod[::-1]

    def _phi(self, r, volume, adjVol, node):
        volumeCentre = self.mb.tag_get_data(
            self.mesh_data.volume_centre_tag, volume
        )
        volumePerm = self.mb.tag_get_data(self.mesh_data.perm_tag, volume)
        adjVolCentre = self.mb.tag_get_data(
            self.mesh_data.volume_centre_tag, adjVol
        )
        adjVolPerm = (self.mb.tag_get_data(self.mesh_data.perm_tag, adjVol),)
        netaVolume_r = self._netaLpew2(r, volumePerm, volumeCentre, node)
        netaVolume_r_plus_1 = self._netaLpew2(
            r + 1, volumePerm, volumeCentre, node
        )
        netaAdjVol_r = self._netaLpew2(r, adjVolPerm, adjVolCentre, node)
        netaAdjVol_r_plus_1 = self._netaLpew2(
            r + 1, adjVolPerm, adjVolCentre, node
        )
        phi = (netaAdjVol_r - netaVolume_r) / (
            netaAdjVol_r_plus_1 - netaVolume_r_plus_1
        )
        # print(netaAdjVol_r, netaVolume_r,netaAdjVol_r_plus_1, netaVolume_r_plus_1)
        return phi

    def _prodPhi(self, rStar, j, volume, adjVols, node):
        rRange = self._rRange(rStar, j + 1)
        phiProd = 1.0
        for r in rRange:
            phi = self._phi(r, volume, adjVols[(r + 1) // 2], node)
            phiProd = phiProd * phi
        return phiProd

    def _netaLpew2(self, r, perm, volumeCentre, node):
        nodeCoords = self.mb.get_coords([node])
        faceVerts = np.asarray(
            [self.T_coords[r], self.T_coords[r + 1], nodeCoords]
        )
        volVerts = np.append(faceVerts, volumeCentre).reshape([4, 3])
        tetraVolume = geo.get_tetra_volume(volVerts)
        N_out = geo._area_vector(faceVerts, self.T_coords[r])[0]
        N_centre = geo._area_vector(faceVerts, volumeCentre[0])[0]
        neta = self._fluxTerm(
            N_out, np.asarray(perm).reshape([3, 3]), N_centre, tetraVolume
        )
        return neta

    def _csi_lpew2(self, volVerts, aux_vert, perm):
        vol_nodes_coords = np.asarray(
            [self.mb.get_coords([vol]) for vol in volVerts]
        )
        aux_vert_coords = self.mb.get_coords([aux_vert])
        vol_nodes_coords.reshape([4, 3])
        tetraVolume = geo.get_tetra_volume(vol_nodes_coords)
        N_node = geo._area_vector(vol_nodes_coords, vol_nodes_coords[3])[0]
        verts_opposite_to_aux_vert = list(set(volVerts).difference({aux_vert}))
        opposite_face_coords = np.asarray(
            [self.mb.get_coords([vol]) for vol in verts_opposite_to_aux_vert]
        )
        N_Tj = geo._area_vector(opposite_face_coords, aux_vert_coords)[0]
        csi = self._fluxTerm(N_node, perm, N_Tj, tetraVolume)
        return csi

    def interpolate(self, node, tao=0.5):
        volsAroundNode = self.mtu.get_bridge_adjacencies(node, 0, 3)
        partialWts = np.zeros([len(volsAroundNode)])
        mapIds = {
            volAroundNode: _id
            for volAroundNode, _id in zip(
                volsAroundNode, range(len(volsAroundNode))
            )
        }
        A = np.zeros([len(volsAroundNode), 6])
        for count, volAroundNode in zip(
            range(len(volsAroundNode)), volsAroundNode
        ):
            print("wer", self.mtu.get_average_position([volAroundNode]))
            verts = self.mtu.get_bridge_adjacencies(volAroundNode, 3, 0)
            coords = []
            for vert in verts:
                coords.append(self.mb.get_coords([vert]))
                print(coords)

            adjVols = self._getVolumesSharingFaceAndNode(node, volAroundNode)
            for id_, adjVol in adjVols.items():

                print("qwerqwer", self.mtu.get_average_position([adjVol]))

            perm = np.asarray(
                self.mb.tag_get_data(self.mesh_data.perm_tag, volAroundNode)
            )
            perm = perm.reshape([3, 3])
            self._getAuxiliaryVerts(node, volAroundNode)
            self._getAuxiliaryVertsCoords(node, volAroundNode, tao)
            print(self.T_coords)
            auxVols = self._getAuxiliaryVolumes(self.T, node)
            phi = self._prodPhi(1, 5, volAroundNode, adjVols, node)
            print(phi)
            for j in range(1, 7):
                A = self._computeA(j, auxVols, perm)
