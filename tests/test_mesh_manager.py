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

        self.mesh_1 = MeshManager('meshes/mesh_test_1.msh')
        self.mesh_2 = MeshManager('meshes/mesh_test_2.msh', dim=3)
        self.all_volumes_2 = self.mesh_2.mb.get_entities_by_dimension(0, 3)
        self.mesh_2 = MeshManager('meshes/mesh_test_2.msh', dim=3)
        self.mesh_2.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_2.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_2.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_2 = MpfaD3D(self.mesh_2)

        self.mesh_3 = MeshManager('meshes/mesh_test_conservative.msh', dim=3)
        self.mesh_3.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_3.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_3.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_3 = MpfaD3D(self.mesh_3)
        self.mesh_4 = MeshManager('meshes/mesh_darlan.msh')
        self.mesh_4.set_global_id()
        self.mesh_4.get_redefine_centre()

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

    @unittest.skip('skip')
    def test_if_node_weighted_calculation_yelds_analytical_solution(self):
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

    @unittest.skip('later')
    def test_volume_centre_is_importing_geo_properly(self):
        for volume in self.mesh_4.all_volumes:
            vol_id = self.mesh_4.mb.tag_get_data(self.mesh_4.global_id_tag, volume)
            vol_coords = self.mesh_4.mb.tag_get_data(self.mesh_4.volume_centre_tag, volume)
            print(vol_id, vol_coords)

    def test_get_vols_sharing_face_and_node(self):
        for node in self.mesh_4.all_nodes:
            vols_around_node = self.mesh_4.mtu.get_bridge_adjacencies(node, 0, 3)
            for vol in vols_around_node:
                T = self.mesh_4._get_auxiliary_verts(node, vol, 0.5)
                
