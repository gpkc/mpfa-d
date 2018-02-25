import unittest
import numpy as np
from pressure_solver_3D import MpfaD3D
from mesh_preprocessor import MeshManager


class PressureSolverTest(unittest.TestCase):

    def setUp(self):

        K_1 = np.array([1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0])

        K_2 = np.array([2.0, 0.0, 0.0,
                        0.0, 2.0, 0.0,
                        0.0, 0.0, 2.0])

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
