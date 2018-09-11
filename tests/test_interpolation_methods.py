import unittest
import numpy as np
from mpfad.MpfaD import MpfaD3D
from mpfad.interpolation.IDW import IDW
from mpfad.interpolation.LSW import LSW
from mpfad.interpolation.LPEW3 import LPEW3
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

        self.mesh_4 = MeshManager('mesh_test_7.msh', dim=3)
        self.mesh_4.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_4.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_4.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_4 = MpfaD3D(self.mesh_4)

        self.mesh_5 = MeshManager('geometry_two_regions_test.msh', dim=3)
        self.mesh_5.set_media_property('Permeability', {1: K_2, 2: K_1}, dim_target=3)
        self.mesh_5.set_boundary_condition('Dirichlet', {102: 1.0, 101: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_5.set_boundary_condition('Neumann', {201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_5 = MpfaD3D(self.mesh_5)

        self.mesh_6 = MeshManager('mesh_test_6.msh', dim=3)
        self.mesh_6.set_media_property('Permeability', {1: K_1}, dim_target=3)
        self.mesh_6.set_boundary_condition('Dirichlet', {102: 1.0},
                                           dim_target=2, set_nodes=True)
        self.mesh_6.set_boundary_condition('Neumann', {202: 1.0, 201: 0.0},
                                           dim_target=2, set_nodes=True)
        self.mpfad_6 = MpfaD3D(self.mesh_6)

    @unittest.skip("we'll see it later")
    def test_inverse_distance_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_inv_distance = IDW(self.mesh_1).interpolate(intern_node)
        for vol, weight in vols_ws_by_inv_distance.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)

    @unittest.skip("we'll see it later")
    def test_least_squares_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_least_squares = LSW(self.mesh_1).interpolate(intern_node)
        for vol, weight in vols_ws_by_least_squares.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)

    @unittest.skip("we'll see it later")
    def test_linear_problem_with_inverse_distance_interpolation_mesh_1(self):
        self.mpfad_1.run_solver(IDW(self.mesh_1).interpolate)
        for a_volume in self.mesh_1.all_volumes:
            local_pressure = self.mesh_1.mb.tag_get_data(
                             self.mpfad_1.pressure_tag, a_volume)
            coord_x = self.mesh_1.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    @unittest.skip("we'll see it later")
    def test_linear_problem_with_least_squares_interpolation_mesh_1(self):
        self.mpfad_1.run_solver(LSW(self.mesh_1).interpolate)
        for a_volume in self.mesh_1.all_volumes:
            local_pressure = self.mesh_1.mb.tag_get_data(
                             self.mpfad_1.pressure_tag, a_volume)
            coord_x = self.mesh_1.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    @unittest.skip("we'll see it later")
    def test_lpew3_yields_same_weight_for_equal_tetrahedra(self):
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_lpew3 = LPEW3(self.mesh_1).interpolate(intern_node)
        for vol, weight in vols_ws_by_lpew3.items():
            self.assertAlmostEqual(weight, 1.0/12.0, delta=1e-15)

    @unittest.skip("we'll see it later")
    def test_linear_problem_with_lpew3_interpolation_mesh_2(self):
        self.mpfad_2.run_solver(LPEW3(self.mesh_2).interpolate)
        for a_volume in self.mesh_2.all_volumes:
            local_pressure = self.mesh_2.mb.tag_get_data(
                             self.mpfad_2.pressure_tag, a_volume)
            coord_x = self.mesh_2.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)
            # print(local_pressure, 1 - coord_x)

    # @unittest.skip("we'll see it later")
    def test_linear_problem_with_neumann_lpew3_interpolation_mesh_4(self):
        self.mpfad_4.run_solver(LPEW3(self.mesh_4).interpolate)
        for a_volume in self.mesh_4.all_volumes:
            local_pressure = self.mesh_4.mb.tag_get_data(
                             self.mpfad_4.pressure_tag, a_volume)
            coord_x = self.mesh_4.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    @unittest.skip("we'll see it later")
    def test_number_of_non_null_neumann_faces(self):
        neumann_faces = self.mpfad_6.neumann_faces
        count = 0

        for neumann_face in neumann_faces:
            flux = self.mesh_6.mb.tag_get_data(
                   self.mpfad_6.neumann_tag, neumann_face)
            if flux == 1.0:
                count += 1
        self.assertEqual(count, 2)

    # @unittest.skip("we'll see it later")
    def test_linear_problem_with_non_null_neumann_condition_lpew3(self):
        self.mpfad_6.run_solver(LPEW3(self.mesh_6).interpolate)
        for a_volume in self.mesh_6.all_volumes:
            local_pressure = self.mesh_6.mb.tag_get_data(
                             self.mpfad_6.pressure_tag, a_volume)
            coord_x = self.mesh_6.get_centroid(a_volume)[0]

            # print(local_pressure, 1 - coord_x, local_pressure/(1 - coord_x))

            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15)

    @unittest.skip("we'll see it later")
    def test_lin_prob_heterog_mesh_lpew3_neumann_intern_nodes_mesh_5(self):
        self.mpfad_5.run_solver(LPEW3(self.mesh_5).interpolate)
        for a_volume in self.mesh_5.all_volumes:
            local_pressure = self.mesh_5.mb.tag_get_data(
                             self.mpfad_5.pressure_tag, a_volume)
            coord_x = self.mesh_5.get_centroid(a_volume)[0]
            if coord_x < 0.5:
                self.assertAlmostEqual(
                    local_pressure[0][0], (-2/3.0)*coord_x + 1, delta=1e-14)
            else:
                self.assertAlmostEqual(
                    local_pressure[0][0], (4/3.0)*(1 - coord_x), delta=1e-14)

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
