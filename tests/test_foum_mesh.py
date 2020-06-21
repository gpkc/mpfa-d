import unittest

from preprocessor.mesh_preprocessor import MeshManager


class TestMeshManagerFoum(unittest.TestCase):
    """Test integration of mesh manager with the two phase solver."""

    def setUp(self):
        self.mesh = MeshManager("meshes/mesh_test_conservative.msh", dim=3)
        sw = 0.2
        so_i = 0.9
        self.mesh.set_media_property("Water_Sat_i", {1: sw}, dim_target=3)
        self.mesh.set_media_property("Oil_Sat_i", {1: so_i}, dim_target=3)
        self.mesh.set_media_property("Water_Sat", {1: sw}, dim_target=3)
        self.mesh.set_boundary_condition(
            "SW_BC", {102: 1.0}, dim_target=2, set_nodes=True
        )

    def test_reading_mesh_return_moab_mesh_instance(self):
        """Test that MeshManager return a moab object containing the mesh."""
        volumes = self.mesh.all_volumes

        self.assertTrue(self.mesh)
        self.assertEqual(len(volumes), 96)

    def test_water_saturation_initial_cond(self):
        """Test setting water saturation Initial Cond at the volumes"""
        water_saturation_tag = self.mesh.water_sat_tag
        volumes = self.mesh.all_volumes
        for volume in volumes:
            water_saturation = self.mesh.mb.tag_get_data(
                water_saturation_tag, volume
            )[0]
            self.assertEqual(water_saturation, 0.2)

    def test_add_water_saturation_BC(self):
        """Test that boundary conditions to the saturation problem is set."""
        water_saturation_bc_tag = self.mesh.water_sat_bc_tag
        for face in self.mesh.sat_BC_faces:
            sw = self.mesh.mb.tag_get_data(water_saturation_bc_tag, face)[0]
            self.assertEqual(sw, 1.0)
