import unittest
import numpy as np
import mpfad.helpers.geometric as geo
from mpfad.MpfaD import MpfaD3D
from mpfad.interpolation.LPEW3 import LPEW3
from mesh_preprocessor import MeshManager
from pymoab import types


class PressureSolverTest(unittest.TestCase):

    """
    These tests are more MeshManager class related. We should rename
    this test suite.
    """
    def setUp(self):

        K_1 = np.array([1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0])

        self.mesh = MeshManager('meshes/mesh_test_conservative.msh', dim=3)
        self.mesh.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                         dim_target=2, set_nodes=True)
        self.mesh.set_boundary_condition('Neumann', {201: 0.0},
                                         dim_target=2, set_nodes=True)
        self.mesh.get_redefine_centre()
        self.mesh.set_global_id()
        self.mpfad = MpfaD3D(self.mesh)

    # @unittest.skip('later')
    def test_if_method_yields_exact_solution(self):
        self.mtu = self.mesh.mtu
        self.mb = self.mesh.mb
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        for volume in self.mesh.all_volumes:
            c_solution = self.mb.tag_get_data(self.mpfad.pressure_tag, volume)
            a_solution = 1 - self.mb.get_coords([volume])[0]
            self.assertAlmostEqual(c_solution, a_solution, delta=5e-15)

    # @unittest.skip('later')
    def test_if_inner_verts_weighted_calculation_yelds_exact_solution(self):
        self.mtu = self.mpfad.mtu
        self.mb = self.mpfad.mb
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        for node in self.mpfad.intern_nodes:
            analytical_solution = 1 - self.mb.get_coords([node])[0]
            nd_weights = self.mpfad.nodes_ws[node]
            p_vert = 0.
            for volume, wt in nd_weights.items():
                p_vol = self.mpfad.mb.tag_get_data(self.mpfad.pressure_tag,
                                                   volume)
                p_vert += p_vol * wt
            self.assertAlmostEqual(p_vert, analytical_solution,
                                   delta=5e-15)

    # @unittest.skip('later')
    def test_if_gradient_yields_correct_values(self):
        self.node_pressure_tag = self.mpfad.mb.tag_get_handle(
            "Node Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
                                                              )
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        p_verts = []
        for node in self.mesh.all_nodes:
            try:
                p_vert = self.mpfad.mb.tag_get_data(self.mpfad.dirichlet_tag,
                                                    node)
                p_verts.append(p_vert[0])
            except:
                p_vert = 0.0
                p_tag = self.mpfad.pressure_tag
                nd_weights = self.mpfad.nodes_ws[node]
                for volume, wt in nd_weights.items():
                    p_vol = self.mpfad.mb.tag_get_data(p_tag, volume)
                    p_vert += p_vol * wt
                p_verts.append(p_vert[0])
        self.mpfad.mb.tag_set_data(self.node_pressure_tag, self.mesh.all_nodes,
                                   p_verts)
        for a_volume in self.mesh.all_volumes:
            vol_faces = self.mesh.mtu.get_bridge_adjacencies(a_volume, 2, 2)
            vol_nodes = self.mesh.mtu.get_bridge_adjacencies(a_volume, 0, 0)
            vol_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_crds = np.reshape(vol_crds, ([4, 3]))
            vol_volume = self.mesh.get_tetra_volume(vol_crds)
            I, J, K = self.mesh.mtu.get_bridge_adjacencies(vol_faces[0], 2, 0)
            L = list(set(vol_nodes).difference(set(
                self.mesh.mtu.get_bridge_adjacencies(vol_faces[0], 2, 0))))
            JI = self.mesh.mb.get_coords([I]) - self.mesh.mb.get_coords([J])
            JK = self.mesh.mb.get_coords([K]) - self.mesh.mb.get_coords([J])
            LJ = self.mesh.mb.get_coords([J]) - self.mesh.mb.get_coords(L)
            N_IJK = np.cross(JI, JK) / 2.

            test = np.dot(LJ, N_IJK)
            if test < 0.:
                I, K = K, I
                JI = self.mesh.mb.get_coords([I]) - \
                    self.mesh.mb.get_coords([J])
                JK = self.mesh.mb.get_coords([K]) - \
                    self.mesh.mb.get_coords([J])
                N_IJK = np.cross(JI, JK) / 2.

            tan_JI = np.cross(N_IJK, JI)
            tan_JK = np.cross(N_IJK, JK)
            face_area = np.sqrt(np.dot(N_IJK, N_IJK))

            h_L = geo.get_height(N_IJK, LJ)

            p_I = self.mpfad.mb.tag_get_data(self.node_pressure_tag, I)
            p_J = self.mpfad.mb.tag_get_data(self.node_pressure_tag, J)
            p_K = self.mpfad.mb.tag_get_data(self.node_pressure_tag, K)
            p_L = self.mpfad.mb.tag_get_data(self.node_pressure_tag, L)
            grad_normal = - 2 * (p_J - p_L) * N_IJK
            grad_cross_I = (p_J - p_I) * ((np.dot(tan_JK, LJ) / face_area ** 2)
                                          * N_IJK - (h_L / (face_area)) *
                                          tan_JK)
            grad_cross_K = (p_K - p_J) * ((np.dot(tan_JI, LJ) / face_area ** 2)
                                          * N_IJK - (h_L / (face_area)) *
                                          tan_JI)

            grad_p = -(1 / (6 * vol_volume)) * (grad_normal +
                                                grad_cross_I +
                                                grad_cross_K)
            for c_grad, a_grad in zip(grad_p[0], -np.array([1., 0., 0.])):
                self.assertAlmostEqual(c_grad, a_grad, delta=1e-14)

    # @unittest.skip('later')
    def test_if_flux_is_conservative_for_all_intern_volumes(self):

        self.node_pressure_tag = self.mpfad.mb.tag_get_handle(
            "Node Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
                                                              )
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        p_verts = []
        for node in self.mesh.all_nodes:
            try:
                p_vert = self.mpfad.mb.tag_get_data(self.mpfad.dirichlet_tag,
                                                    node)
                p_verts.append(p_vert[0])
            except:
                p_vert = 0.0
                p_tag = self.mpfad.pressure_tag
                nd_weights = self.mpfad.nodes_ws[node]
                for volume, wt in nd_weights.items():
                    p_vol = self.mpfad.mb.tag_get_data(p_tag, volume)
                    p_vert += p_vol * wt
                p_verts.append(p_vert[0])
        self.mpfad.mb.tag_set_data(self.node_pressure_tag, self.mesh.all_nodes,
                                   p_verts)
        for a_volume in self.mesh.all_volumes:
            vol_centroid = self.mesh.mtu.get_average_position([a_volume])
            vol_faces = self.mesh.mtu.get_bridge_adjacencies(a_volume, 2, 2)
            vol_p = self.mesh.mb.tag_get_data(
                    self.mpfad.pressure_tag, a_volume)[0][0]
            vol_nodes = self.mesh.mtu.get_bridge_adjacencies(a_volume, 0, 0)
            vol_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_crds = np.reshape(vol_crds, ([4, 3]))
            vol_volume = self.mesh.get_tetra_volume(vol_crds)
            vol_perm = self.mesh.mb.tag_get_data(
                       self.mpfad.perm_tag, a_volume).reshape([3, 3])
            fluxes = []
            for a_face in vol_faces:
                f_nodes = self.mesh.mtu.get_bridge_adjacencies(a_face, 0, 0)
                fc_nodes = self.mesh.mb.get_coords(f_nodes)
                fc_nodes = np.reshape(fc_nodes, ([3, 3]))
                grad = np.zeros(3)
                for i in range(len(fc_nodes)):
                    area_vect = geo._area_vector(np.array([fc_nodes[i],
                                                          fc_nodes[i-1],
                                                          vol_centroid]),
                                                 fc_nodes[i-2])[0]
                    p_node_op = self.mpfad.mb.tag_get_data(
                                self.node_pressure_tag, f_nodes[i-2])[0][0]
                    grad += area_vect * p_node_op
                area_vect = geo._area_vector(fc_nodes, vol_centroid)[0]
                grad += area_vect * vol_p
                grad = grad / (3.0 * vol_volume)
                flux = - np.dot(np.dot(vol_perm, grad), area_vect)
                fluxes.append(flux)
            fluxes_sum = abs(sum(fluxes))
            self.assertAlmostEqual(fluxes_sum, 0.0, delta=1e-14)
