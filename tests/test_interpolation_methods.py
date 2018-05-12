import unittest
import numpy as np
from pressure_solver_3D import MpfaD3D
from interpolation_methods import InterpolMethod
from mesh_preprocessor import MeshManager


class InterpMethodTest(unittest.TestCase):

    def setUp(self):

        K_1 = np.array([1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0])

        K_2 = np.array([2.0, 0.0, 0.0,
                        0.0, 2.0, 0.0,
                        0.0, 0.0, 2.0])

        self.mesh_1 = MeshManager('mesh_test_1.msh', dim=3)
        self.mesh_1.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_1.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_1.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_1 = MpfaD3D(self.mesh_1)
        self.imd_1 = InterpolMethod(self.mesh_1)

        self.mesh_2 = MeshManager('mesh_test_2.msh', dim=3)
        self.mesh_2.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_2.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_2.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_2 = MpfaD3D(self.mesh_2)
        self.imd_2 = InterpolMethod(self.mesh_2)

    def test_inverse_distance_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_inv_distance = self.imd_1.by_inverse_distance(intern_node)
        for vol, weight in vols_ws_by_inv_distance.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)

    def test_least_squares_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_least_squares = self.imd_1.by_least_squares(intern_node)
        for vol, weight in vols_ws_by_least_squares.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)

    @unittest.skip("not ready for testing")
    def test_lpew2_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_lpew2 = self.imd_1.by_lpew2(intern_node)
        for vol, weight in vols_ws_by_lpew2.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)

    def test_linear_problem_with_inverse_distance_interpolation_mesh_1(self):
        self.mpfad_1.run_solver(self.imd_1.by_inverse_distance)
        for a_volume in self.mesh_1.all_volumes:
            local_pressure = self.mesh_1.mb.tag_get_data(
                             self.mpfad_1.pressure_tag, a_volume)
            coord_x = self.mesh_1.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    def test_linear_problem_with_least_squares_interpolation_mesh_1(self):
        self.mpfad_1.run_solver(self.imd_1.by_least_squares)
        for a_volume in self.mesh_1.all_volumes:
            local_pressure = self.mesh_1.mb.tag_get_data(
                             self.mpfad_1.pressure_tag, a_volume)
            coord_x = self.mesh_1.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    def test_lpew3_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_least_squares = self.imd_1.by_lpew3(intern_node)
        for vol, weight in vols_ws_by_least_squares.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)

    def test_linear_problem_with_lpew3_interpolation_mesh_2(self):
        self.mpfad_2.run_solver(self.imd_2.by_lpew3)
        for a_volume in self.mesh_2.all_volumes:
            local_pressure = self.mesh_2.mb.tag_get_data(
                             self.mpfad_2.pressure_tag, a_volume)
            coord_x = self.mesh_2.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)
