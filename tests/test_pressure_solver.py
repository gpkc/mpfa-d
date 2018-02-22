import unittest
import numpy as np
from pressure_solver_3D import MpfaD3D
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

        self.mesh_1 = MeshManager('geometry_3D_test.msh', dim=3)
        self.all_volumes_1 = self.mesh_1.mb.get_entities_by_dimension(0, 3)
        self.mesh_1.set_media_property('Permeability', {1:K_1})
        self.mesh_1.set_boundary_condition('Dirichlet', {102:1.0, 101:0.0}, True)
        self.mesh_1.set_boundary_condition('Neumann', {201:0.0}, True)
        self.mpfad_1 = MpfaD3D(self.mesh_1)

        self.mesh_2 = MeshManager('geometry_two_regions_test.msh', dim=3)
        self.all_volumes_2 = self.mesh_2.mb.get_entities_by_dimension(0, 3)
        self.mesh_2.set_media_property('Permeability', {1:K_1, 2:K_2})
        self.mesh_2.set_boundary_condition('Dirichlet', {102:1.0, 101:0.0}, True)
        self.mesh_2.set_boundary_condition('Neumann', {201:0.0}, True)
        self.mpfad_2 = MpfaD3D(self.mesh_2)

        self.mesh_3 = MeshManager('geometry_structured_test.msh', dim=3)
        self.all_volumes_2 = self.mesh_3.mb.get_entities_by_dimension(0, 3)
        self.mesh_3.set_media_property('Permeability', {1:K_1})
        self.mesh_3.set_boundary_condition('Dirichlet', {102:1.0, 101:0.0}, True)
        self.mesh_3.set_boundary_condition('Neumann', {201:0.0}, True)
        self.mpfad_3 = MpfaD3D(self.mesh_3)

    def test_if_method_has_all_dirichlet_nodes(self):
        self.assertEqual(len(self.mpfad_2.dirichlet_nodes), 10)
        self.assertEqual(len(self.mpfad_3.dirichlet_nodes), 162)

    def test_if_method_has_all_neumann_nodes(self):
        self.assertEqual(len(self.mpfad_2.neumann_nodes), 12)
        self.assertEqual(len(self.mpfad_3.neumann_nodes), 224)

    def test_if_method_has_all_intern_nodes(self):
        self.assertEqual(len(self.mpfad_2.intern_nodes), 1)
        self.assertEqual(len(self.mpfad_3.intern_nodes), 343)

    def test_if_method_has_all_dirichlet_faces(self):
        self.assertEqual(len(self.mpfad_2.dirichlet_faces), 8)
        self.assertEqual(len(self.mpfad_3.dirichlet_faces), 128)

    def test_if_method_has_all_neumann_faces(self):
        self.assertEqual(len(self.mpfad_2.neumann_faces), 32)
        self.assertEqual(len(self.mpfad_3.neumann_faces), 256)

    def test_if_method_has_all_intern_faces(self):
        self.assertEqual(len(self.mpfad_2.intern_faces), 76)
        self.assertEqual(len(self.mpfad_3.intern_faces), 1344)

        
    # def test_linear_problem_with_orthogonal_mesh(self):
    #     mpfad = MpfaD3D(self.mesh_1)
    #     mpfad.run()
    #     pressure_tag = self.mesh_1.mb.tag_get_handle('Pressure')
    #     self.
    #     self.assertEqual(len(mpfad.dirichlet_nodes), 10)
