import unittest

from preprocessor.mesh_preprocessor import MeshManager
from solvers.foum import Foum


class TestFoumTags(unittest.TestCase):
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
        self.foum = Foum(self.mesh, 1.0, 0.9, 1.0, 1.3, 0.5)

    def test_foum_is_instanciated(self):
        """Test that foum is initiated successfuly."""
        self.assertIsNotNone(self.foum)

    def test_mount_mobility(self):
        self.foum.init()
        for face in self.mesh.all_faces:
            face_mobility = self.mesh.mb.tag_get_data(
                self.mesh.face_mobility_tag, face
            )
            self.assertTrue(face_mobility)
