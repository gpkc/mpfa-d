"""Test the interpolation methods."""
import unittest
import numpy as np
from solvers.MpfaD import MpfaD3D

# from mpfad.interpolation.IDW import IDW
# from mpfad.interpolation.LSW import LSW
from solvers.interpolation.LPEW3 import LPEW3

from preprocessor.mesh_preprocessor import MeshManager


class InterpolationMethodTest(unittest.TestCase):
    """Test suite for interpolation."""

    def setUp(self):
        """Init test suite."""
        K_1 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

        K_2 = np.array([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0])

        self.mesh_1 = MeshManager("meshes/mesh_test_1.msh", dim=3)
        self.mesh_1.set_media_property("Permeability", {1: K_1}, dim_target=3)
        self.mesh_1.set_boundary_condition(
            "Dirichlet", {102: 1.0, 101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_1.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_1.get_redefine_centre()
        self.mpfad_1 = MpfaD3D(self.mesh_1)

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

        self.mesh_4 = MeshManager("meshes/mesh_test_7.msh", dim=3)
        self.mesh_4.set_media_property("Permeability", {1: K_1}, dim_target=3)
        self.mesh_4.set_boundary_condition(
            "Dirichlet", {102: 1.0, 101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_4.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_4.set_global_id()
        self.mesh_4.get_redefine_centre()
        self.mpfad_4 = MpfaD3D(self.mesh_4)

        self.mesh_5 = MeshManager(
            "meshes/geometry_two_regions_test.msh", dim=3
        )
        self.mesh_5.set_media_property(
            "Permeability", {1: K_2, 2: K_1}, dim_target=3
        )
        self.mesh_5.set_boundary_condition(
            "Dirichlet", {102: 1.0, 101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_5.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_5.get_redefine_centre()
        self.mesh_5.set_global_id()
        self.mpfad_5 = MpfaD3D(self.mesh_5)

        self.mesh_6 = MeshManager("meshes/mesh_test_6.msh", dim=3)
        self.mesh_6.set_media_property("Permeability", {1: K_1}, dim_target=3)
        self.mesh_6.set_boundary_condition(
            "Dirichlet", {102: 1.0}, dim_target=2, set_nodes=True
        )
        self.mesh_6.set_boundary_condition(
            "Neumann", {202: 1.0, 201: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_6.get_redefine_centre()
        self.mesh_6.set_global_id()
        self.mpfad_6 = MpfaD3D(self.mesh_6)

    def test_lpew3_yields_same_weight_for_equal_tetrahedra(self):
        """Test if lpew3 interpolation yields correct weights."""
        intern_node = self.mesh_1.all_nodes[-1]
        vols_ws_by_lpew3 = LPEW3(self.mesh_1).interpolate(intern_node)
        for vol, weight in vols_ws_by_lpew3.items():
            self.assertAlmostEqual(weight, 1.0 / 12.0, delta=1e-15)

    def test_linear_problem_with_lpew3_interpolation_mesh_2(self):
        """Test if linear solution is obtained for mesh 2 testcase."""
        self.mpfad_2.run_solver(LPEW3(self.mesh_2).interpolate)
        for a_volume in self.mesh_2.all_volumes:
            local_pressure = self.mesh_2.mb.tag_get_data(
                self.mpfad_2.pressure_tag, a_volume
            )
            coord_x = self.mesh_2.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15
            )

    def test_linear_problem_with_neumann_lpew3_interpolation_mesh_4(self):
        """Test solution with Neumann BC lpew3 interpolation."""
        self.mpfad_4.run_solver(LPEW3(self.mesh_4).interpolate)
        for a_volume in self.mesh_4.all_volumes:
            local_pressure = self.mesh_4.mb.tag_get_data(
                self.mpfad_4.pressure_tag, a_volume
            )
            coord_x = self.mesh_4.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15
            )

    def test_heterogeneous_mesh_lpew3_neumann_intern_nodes_mesh_5(self):
        """Test solution for Neumann BC lpew3 and heterogeneus mesh."""
        self.mpfad_4.run_solver(LPEW3(self.mesh_4).interpolate)
        self.mpfad_5.run_solver(LPEW3(self.mesh_5).interpolate)
        for a_volume in self.mesh_5.all_volumes:
            local_pressure = self.mesh_5.mb.tag_get_data(
                self.mpfad_5.pressure_tag, a_volume
            )
            coord_x = self.mesh_5.get_centroid(a_volume)[0]

            if coord_x < 0.5:
                self.assertAlmostEqual(
                    local_pressure[0][0], (-2 / 3.0) * coord_x + 1, delta=1e-14
                )
            else:
                self.assertAlmostEqual(
                    local_pressure[0][0],
                    (4 / 3.0) * (1 - coord_x),
                    delta=1e-14,
                )

    def test_number_of_non_null_neumann_faces(self):
        """Test if len of neumann faces is not None."""
        neumann_faces = self.mpfad_6.neumann_faces
        count = 0

        for neumann_face in neumann_faces:
            flux = self.mesh_6.mb.tag_get_data(
                self.mpfad_6.neumann_tag, neumann_face
            )
            if flux == 1.0:
                count += 1
        self.assertEqual(count, 2)

    def test_linear_problem_with_non_null_neumann_condition_lpew3(self):
        """Test linear problem with non null neumann BC and lpew3."""
        self.mpfad_6.run_solver(LPEW3(self.mesh_6).interpolate)
        for a_volume in self.mesh_6.all_volumes:
            local_pressure = self.mesh_6.mb.tag_get_data(
                self.mpfad_6.pressure_tag, a_volume
            )
            coord_x = self.mesh_6.get_centroid(a_volume)[0]
            self.assertAlmostEqual(
                local_pressure[0][0], 1 - coord_x, delta=1e-15
            )
