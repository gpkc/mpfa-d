import unittest
import numpy as np
from mpfad.MpfaD import MpfaD3D
from mpfad.interpolation.IDW import IDW
from mpfad.interpolation.LSW import LSW
from mpfad.interpolation.LPEW3 import LPEW3
from mpfad.interpolation.InterpolationMethod import InterpolationMethodBase
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

        self.mesh_2 = MeshManager('mesh_test_2.msh', dim=3)
        self.mesh_2.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_2.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_2.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_2 = MpfaD3D(self.mesh_2)

        self.mesh_3 = MeshManager('mesh_test_3.msh', dim=3)
        self.mesh_3.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_3.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_3.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_3 = MpfaD3D(self.mesh_3)

    def test_inverse_distance_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_inv_distance = IDW(self.mesh_1).interpolate(intern_node)
        for vol, weight in vols_ws_by_inv_distance.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)

    def test_least_squares_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_least_squares = LSW(self.mesh_1).interpolate(intern_node)
        for vol, weight in vols_ws_by_least_squares.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)

    def test_linear_problem_with_inverse_distance_interpolation_mesh_1(self):
        self.mpfad_1.run_solver(IDW(self.mesh_1).interpolate)
        for a_volume in self.mesh_1.all_volumes:
            local_pressure = self.mesh_1.mb.tag_get_data(
                             self.mpfad_1.pressure_tag, a_volume)
            coord_x = self.mesh_1.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    def test_linear_problem_with_least_squares_interpolation_mesh_1(self):
        self.mpfad_1.run_solver(LSW(self.mesh_1).interpolate)
        for a_volume in self.mesh_1.all_volumes:
            local_pressure = self.mesh_1.mb.tag_get_data(
                             self.mpfad_1.pressure_tag, a_volume)
            coord_x = self.mesh_1.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    def test_lpew3_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_lpew3 = LPEW3(self.mesh_1).interpolate(intern_node)
        for vol, weight in vols_ws_by_lpew3.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)

    def test_linear_problem_with_lpew3_interpolation_mesh_2(self):
        self.mpfad_2.run_solver(LPEW3(self.mesh_2).interpolate)
        for a_volume in self.mesh_2.all_volumes:
            local_pressure = self.mesh_2.mb.tag_get_data(
                             self.mpfad_2.pressure_tag, a_volume)
            coord_x = self.mesh_2.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    @unittest.skip("we'll see it later")
    def test_if_get_node_by_MeshManager_yields_same_as_in_InterpolMethod(self):
        # this is a mesh manager test
        intern_node = self.imd_3.intern_nodes.pop()
        self.assertTrue(intern_node == self.mesh_3.all_nodes[0])

    @unittest.skip("not ready for testing. mesh seems to be corrupted")
    def test_if_support_region_for_lpew2_is_correct(self):
        from pymoab import core
        from pymoab import topo_util

        mb = core.Core()
        mb.load_file('mesh_test_3.msh')
        mb.write_file('mesh_test_3_.msh')
        root_set = mb.get_root_set()
        mtu = topo_util.MeshTopoUtil(mb)

        all_nodes = mb.get_entities_by_dimension(0, 0)

        mtu.construct_aentities(all_nodes)

        nodes = mb.get_entities_by_dimension(0, root_set)

        for node in nodes:
            adj_vols = list(mtu.get_bridge_adjacencies(node, 3, 3))
            if len(adj_vols) == 4:
                for vol in adj_vols:
                    pass

    @unittest.skip("not ready for testing")
    def test_lpew2_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_lpew2 = self.imd_1.by_lpew2(intern_node)
        for vol, weight in vols_ws_by_lpew2.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)
