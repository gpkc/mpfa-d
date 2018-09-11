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

        self.mesh_3 = MeshManager('mesh_test_conservative.msh', dim=3)
        self.mesh_3.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_3.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_3.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_3 = MpfaD3D(self.mesh_3)

    #@unittest.skip('later')
    def test_if_method_yields_exact_solution(self):
        self.mtu = self.mpfad_3.mtu
        self.mb = self.mpfad_3.mb
        self.mpfad_3.run_solver(LPEW3(self.mesh_3).interpolate)
        for volume in self.mesh_3.all_volumes:
            calc_solution = self.mb.tag_get_data(self.mpfad_3.pressure_tag,
                                                 volume)
            analytical_solution = 1 - self.mb.get_coords([volume])[0]
            self.assertAlmostEqual(calc_solution, analytical_solution,
                                   delta=5e-15)

<<<<<<< HEAD
    #@unittest.skip('later')
=======
    # @unittest.skip('later')
>>>>>>> b0ab657cc22c340851909bb5770b0f7a3d986fce
    def test_if_inner_verts_weighted_calculation_yelds_exact_solution(self):
        self.mtu = self.mpfad_3.mtu
        self.mb = self.mpfad_3.mb
        self.mpfad_3.run_solver(LPEW3(self.mesh_3).interpolate)
        for node in self.mpfad_3.intern_nodes:
            analytical_solution = 1 - self.mb.get_coords([node])[0]
            nd_weights = LPEW3(self.mesh_3).interpolate(node)
            p_vert = 0.
            for volume, wt in nd_weights.items():
                p_vol = self.mpfad_3.mb.tag_get_data(self.mpfad_3.pressure_tag,
                                                     volume)
                p_vert += p_vol * wt
            self.assertAlmostEqual(p_vert, analytical_solution,
                                   delta=5e-15)
    @unittest.skip('later')
    def test_if_flux_is_conservative_for_all_volumes(self):

        self.node_pressure_tag = self.mpfad_3.mb.tag_get_handle(
            "Node Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
                                                                )
        self.mpfad_3.run_solver(LPEW3(self.mesh_3).interpolate)
        inner_volumes = self.mesh_3.get_non_boundary_volumes(
                        self.mpfad_3.dirichlet_nodes,
                        self.mpfad_3.neumann_nodes)
        self.node_pressure_tag = self.mpfad_3.mb.tag_get_handle(
            "Node Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
                                                                )
        for node in self.mpfad_3.dirichlet_nodes:
            self.mpfad_3.mb.tag_set_data(self.node_pressure_tag, node,
                                         self.mpfad_3.mb.tag_get_data(
                                            self.mpfad_3.dirichlet_tag, node)
                                         )

        for node in self.mpfad_3.intern_nodes:
            nd_weights = LPEW3(self.mesh_3).interpolate(node)
            p_vert = 0.0
            for volume, wt in nd_weights.items():
                p_vol = self.mpfad_3.mb.tag_get_data(self.mpfad_3.pressure_tag,
                                                     volume)
                p_vert += p_vol * wt
            self.mpfad_3.mb.tag_set_data(self.node_pressure_tag, node, p_vert)

        for a_volume in inner_volumes:
            vol_centroid = self.mesh_3.mtu.get_average_position([a_volume])
            vol_faces = self.mesh_3.mtu.get_bridge_adjacencies(a_volume, 2, 2)
            vol_p = self.mesh_3.mb.tag_get_data(
                    self.mpfad_3.pressure_tag, a_volume)[0][0]
            # print('v: ', vol_p)
            vol_nodes = self.mesh_3.mtu.get_bridge_adjacencies(a_volume, 0, 0)
            vol_crds = self.mesh_3.mb.get_coords(vol_nodes)
            vol_crds = np.reshape(vol_crds, ([4, 3]))
            vol_volume = self.mesh_3.get_tetra_volume(vol_crds)
            vol_perm = self.mesh_3.mb.tag_get_data(
                       self.mpfad_3.perm_tag, a_volume).reshape([3, 3])
            fluxes = []
            for a_face in vol_faces:
                f_nodes = self.mesh_3.mtu.get_bridge_adjacencies(a_face, 0, 0)
                fc_nodes = self.mesh_3.mb.get_coords(f_nodes)
                fc_nodes = np.reshape(fc_nodes, ([3, 3]))
                grad = np.zeros(3)
                for i in range(len(fc_nodes)):
                    area_vect = geo._area_vector(np.array([fc_nodes[i],
                                                          fc_nodes[i-1],
                                                          vol_centroid]),
                                                 fc_nodes[i-2])[0]
                    p_node_op = self.mpfad_3.mb.tag_get_data(
                                self.node_pressure_tag, f_nodes[i-2])[0][0]

                    grad += area_vect * p_node_op

                area_vect = geo._area_vector(fc_nodes, vol_centroid)[0]
                grad += area_vect * vol_p
                grad = grad / (3.0 * vol_volume)

                flux = - np.dot(np.dot(vol_perm, grad), area_vect)
                fluxes.append(flux)
            fluxes_sum = abs(sum(fluxes))
            self.assertAlmostEqual(fluxes_sum, 0.0, delta=1e-14)


    @unittest.skip('later')
    def test_if_flux_is_conservative_for_non_boundary_volumes(self):
        self.mtu = self.mpfad_3.mtu
        self.mb = self.mpfad_3.mb
        inner_volumes = self.mesh_3.get_non_boundary_volumes(
                        self.mpfad_3.dirichlet_nodes,
                        self.mpfad_3.neumann_nodes)
        nd_weights = self.mpfad_3.run_solver(LPEW3(self.mesh_3).interpolate)

        # self.mpfad_3.record_data('conservative_test.vtk')
        self.node_pressure_tag = self.mpfad_3.mb.tag_get_handle(
            "Node Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
                                                                )
        for node in self.mpfad_3.dirichlet_nodes:
            self.mpfad_3.mb.tag_set_data(self.node_pressure_tag, node,
                                         self.mpfad_3.mb.tag_get_data(
                                             self.mpfad_3.dirichlet_tag, node)
                                         )

        for node in self.mpfad_3.intern_nodes:
            nd_weights = LPEW3(self.mesh_3).interpolate(node)
            p_vert = 0.0
            for volume, wt in nd_weights.items():
                p_vol = self.mpfad_3.mb.tag_get_data(self.mpfad_3.pressure_tag,
                                                     volume)
                p_vert += p_vol * wt
            self.mpfad_3.mb.tag_set_data(self.node_pressure_tag, node, p_vert)

        for volume in inner_volumes:

            # vol_nodes = self.mb.get_adjacencies(volume, 0)
            # for v_node in vol_nodes:
            #     print('TEST NODE:', v_node in self.mpfad_3.neumann_nodes)
            #     print('TEST NODE:', v_node in self.mpfad_3.dirichlet_nodes)

            fl = []
            u = self.mb.tag_get_data(self.mpfad_3.pressure_tag, volume)
            all_faces = self.mtu.get_bridge_adjacencies(volume, 3, 2)
            volume_centroid = self.mtu.get_average_position([volume])
            for face in all_faces:
                I, J, K = self.mtu.get_bridge_adjacencies(face, 0, 0)
                g_I = self.mb.tag_get_data(self.node_pressure_tag, I)
                g_J = self.mb.tag_get_data(self.node_pressure_tag, J)
                g_K = self.mb.tag_get_data(self.node_pressure_tag, K)
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                face_centroid = self.mtu.get_average_position([face])

                test_vector = face_centroid - volume_centroid
                N_IJK = self.mpfad_3.area_vector(JK, JI, test_vector)
                face_area = np.sqrt(np.dot(N_IJK, N_IJK))

                JR = volume_centroid - self.mb.get_coords([J])
                h_R = np.absolute(np.dot(N_IJK, JR) / np.sqrt(np.dot(N_IJK,
                                  N_IJK)))

                tan_JI = np.cross(N_IJK, JI)
                tan_JK = np.cross(JK, N_IJK)

                K_R = self.mb.tag_get_data(self.mpfad_3.perm_tag,
                                           volume).reshape([3, 3])

                K_R_n = self.mpfad_3._flux_term(N_IJK, K_R, N_IJK, face_area)
                K_R_JI = self.mpfad_3._flux_term(N_IJK, K_R, -tan_JI, face_area)
                K_R_JK = self.mpfad_3._flux_term(N_IJK, K_R, -tan_JK, face_area)

                D_JI = self.mpfad_3._boundary_cross_term(tan_JK, JR, face_area,
                                                         K_R_JK, K_R_n, h_R)
                D_JK = self.mpfad_3._boundary_cross_term(tan_JI, JR, face_area,
                                                         K_R_JI, K_R_n, h_R)
                K_n_eff = K_R_n / h_R
                flux = -(2 * K_n_eff * (u - g_J) +
                         D_JI * (g_J - g_I) +
                         D_JK * (g_J - g_K))[0][0]
                print('FLUXES: ', volume_centroid, flux)
                fl.append(flux)
            total_flux = sum(fl)
            print('total_flux', total_flux)
            self.assertAlmostEqual(sum(fl), 0.0, delta=1e-15)
