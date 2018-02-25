import unittest
import numpy as np
from pressure_solver_3D import MpfaD3D
from pressure_solver_3D import InterpolMethod
from mesh_preprocessor import MeshManager
from pymoab import types
from unittest import mock


class PressureSolverTest(unittest.TestCase):

    def setUp(self):

        K_1 = np.array([1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0])

        K_2 = np.array([2.0, 0.0, 0.0,
                        0.0, 2.0, 0.0,
                        0.0, 0.0, 2.0])

        self.mesh_1 = MeshManager('mesh_test_1.msh', dim=3)
        self.all_volumes_1 = self.mesh_1.mb.get_entities_by_dimension(0, 3)
        self.mesh_1.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_1.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_1.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_1 = MpfaD3D(self.mesh_1)
        self.inverse_dist_inter = InterpolMethod(self.mesh_1)

        self.mesh_2 = MeshManager('geometry_two_regions_test.msh', dim=3)
        self.all_volumes_2 = self.mesh_2.mb.get_entities_by_dimension(0, 3)
        self.mesh_2.set_media_property('Permeability', {1: K_1, 2: K_2},
                                       dim_target=3)
        self.mesh_2.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_2.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_2 = MpfaD3D(self.mesh_2)

    def test_if_method_has_all_dirichlet_nodes(self):
        self.assertEqual(len(self.mpfad_2.dirichlet_nodes), 10)

    def test_if_method_has_all_neumann_nodes(self):
        self.assertEqual(len(self.mpfad_2.neumann_nodes), 12)

    def test_if_method_has_all_intern_nodes(self):
        self.assertEqual(len(self.mpfad_2.intern_nodes), 1)

    def test_if_method_has_all_dirichlet_faces(self):
        self.assertEqual(len(self.mpfad_2.dirichlet_faces), 8)

    def test_if_method_has_all_neumann_faces(self):
        self.assertEqual(len(self.mpfad_2.neumann_faces), 32)

    def test_if_method_has_all_intern_faces(self):
        self.assertEqual(len(self.mpfad_2.intern_faces), 76)

    def test_linear_problem_with_no_interpolation(self):
        self.mpfad_1.run()
        for a_volume in self.mesh_1.all_volumes:
            local_pressure = self.mesh_1.mb.tag_get_data(
                             self.mpfad_1.pressure_tag, a_volume)
            coord_x = self.mesh_1.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-10)

    def test_inverse_distance_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        print('NODE COORD', self.mesh_1.mb.get_coords([intern_node]))
        vols_weights = self.inverse_dist_inter.by_inverse_distance(intern_node)
        for vol, weight in vols_weights.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-10)
