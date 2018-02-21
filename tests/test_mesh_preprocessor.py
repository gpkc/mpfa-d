import unittest
from MPFAD_3D.pressure_solver_3D import MpfaD3D
from MPFAD_3D.mesh_processor import MeshManager
from unittest import mock

class MeshManagerTest(unittest.TestCase):

    def setUp(self):

        self.mesh = MeshManager('geometry_two_regions.msh')

        self.K1 = np.array([2.0, 0.0, 0.0,
                            0.0, 2.0, 0.0,
                            0.0, 0.0, 2.0])

        self.K2 = np.array([2.0, 0.0, 0.0,
                            0.0, 2.0, 0.0,
                            0.0, 0.0, 2.0])

    def test_permeability_is_being_attribuited_to_the_elements(self):
        with mock.patch('MPFAD_3D.mesh_processor.mb.tag_set_data') as mock_tag_set:
            self.mesh.media_property({1:self.K1, 2:self.K2})
            self.assertEqual(mock_tag_set.call_count, 48)
