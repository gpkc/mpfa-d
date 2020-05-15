import unittest
import numpy as np

from solvers.MpfaD import MpfaD3D
from preprocessor.mesh_preprocessor import MeshManager


class MeshManagerTest(unittest.TestCase):

    """
    These tests are more MeshManager class related. We should rename
    this test suite.
    """

    def setUp(self):

        K_1 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        K_2 = np.array([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0])

        self.mesh_1 = MeshManager("meshes/mesh_test_1.msh")

        self.mesh_2 = MeshManager("meshes/mesh_test_2.msh", dim=3)
        self.mesh_2.set_media_property("Permeability", {1: K_1}, dim_target=3)
        self.mesh_2.set_boundary_condition(
            "Dirichlet", {102: 1.0, 101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_2.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_2.set_global_id()
        self.mesh_2.get_redefine_centre()
        self.mpfad_2 = MpfaD3D(self.mesh_2)

        self.mesh_3 = MeshManager(
            "meshes/geometry_two_regions_test.msh", dim=3
        )
        self.mesh_3.set_media_property(
            "Permeability", {1: K_1, 2: K_2}, dim_target=3
        )
        self.mesh_3.set_boundary_condition(
            "Dirichlet", {102: 1.0, 101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_3.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_3.get_redefine_centre()
        self.mpfad_3 = MpfaD3D(self.mesh_3)

        self.mesh_4 = MeshManager("meshes/mesh_darlan.msh")
        self.mesh_4.set_global_id()
        self.mesh_4.get_redefine_centre()

    def test_if_method_has_all_dirichlet_nodes(self):
        self.assertEqual(len(self.mpfad_3.dirichlet_nodes), 10)

    def test_if_method_has_all_neumann_nodes(self):
        self.assertEqual(len(self.mpfad_3.neumann_nodes), 12)

    def test_if_method_has_all_intern_nodes(self):
        self.assertEqual(len(self.mpfad_3.intern_nodes), 1)

    def test_if_method_has_all_dirichlet_faces(self):
        self.assertEqual(len(self.mpfad_2.dirichlet_faces), 4)

    def test_if_method_has_all_neumann_faces(self):
        self.assertEqual(len(self.mpfad_2.neumann_faces), 8)

    def test_if_method_has_all_intern_faces(self):
        self.assertEqual(len(self.mpfad_2.intern_faces), 18)

    def test_get_tetra_volume(self):
        a_tetra = self.mesh_1.all_volumes[0]
        tetra_nodes = self.mesh_1.mb.get_adjacencies(a_tetra, 0)
        tetra_nodes_coords = self.mesh_1.mb.get_coords(tetra_nodes)
        tetra_nodes_coords = np.reshape(tetra_nodes_coords, (4, 3))
        vol_eval = self.mesh_1.get_tetra_volume(tetra_nodes_coords)
        self.assertAlmostEqual(vol_eval, 1 / 12.0, delta=1e-15)
