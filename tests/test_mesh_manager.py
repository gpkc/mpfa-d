import unittest
import numpy as np
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

        self.mesh_1 = MeshManager('mesh_test_1.msh')
        self.mesh_2 = MeshManager('mesh_test_2.msh', dim=3)
        self.all_volumes_2 = self.mesh_2.mb.get_entities_by_dimension(0, 3)
        self.mesh_2 = MeshManager('mesh_test_2.msh', dim=3)
        self.mesh_2.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_2.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_2.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_2 = MpfaD3D(self.mesh_2)

        self.mesh_3 = MeshManager('mesh_test_conservative.msh', dim=3)
        self.mesh_3.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_3.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_3.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_3 = MpfaD3D(self.mesh_3)

    @unittest.skip('no need for testing')
    def test_if_method_has_all_dirichlet_nodes(self):
        self.assertEqual(len(self.mpfad_2.dirichlet_nodes), 10)

    @unittest.skip('no need for testing')
    def test_if_method_has_all_neumann_nodes(self):
        self.assertEqual(len(self.mpfad_2.neumann_nodes), 12)

    @unittest.skip('no need for testing')
    def test_if_method_has_all_intern_nodes(self):
        self.assertEqual(len(self.mpfad_2.intern_nodes), 1)

    @unittest.skip('no need for testing')
    def test_if_method_has_all_dirichlet_faces(self):
        self.assertEqual(len(self.mpfad_2.dirichlet_faces), 8)

    @unittest.skip('no need for testing')
    def test_if_method_has_all_neumann_faces(self):
        self.assertEqual(len(self.mpfad_2.neumann_faces), 32)

    @unittest.skip('no need for testing')
    def test_if_method_has_all_intern_faces(self):
        self.assertEqual(len(self.mpfad_2.intern_faces), 76)

    @unittest.skip('no need for testing')
    def test_get_tetra_volume(self):
        a_tetra = self.mesh_1.all_volumes[0]
        tetra_nodes = self.mesh_1.mb.get_adjacencies(a_tetra, 0)
        tetra_nodes_coords = self.mesh_1.mb.get_coords(tetra_nodes)
        tetra_nodes_coords = np.reshape(tetra_nodes_coords, (4, 3))
        vol_eval = self.mesh_1.get_tetra_volume(tetra_nodes_coords)
        self.assertAlmostEqual(vol_eval, 1/12.0, delta=1e-15)

    def test_if_node_weighted_calculation_yelds_analytical_solution(self):
        self.mtu = self.mpfad_3.mtu
        self.mb = self.mpfad_3.mb
        # inner_volumes = self.mesh_3.get_non_boundary_volumes(
        #                 self.mpfad_3.dirichlet_nodes,
        #                 self.mpfad_3.neumann_nodes)
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
