import unittest
from pressure_solver_3D import MpfaD3D
from mesh_processor import MeshManager
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

        self.mesh_1 = MeshManager('geometry_3D_test.msh', dimension=3)
        self.all_volumes_1 = self.mesh_1.mb.get_entities_by_dimension(0, 3)
        self.mesh_1.set_media_property('Permeability', {1:K_1})
        self.mesh_1.set_boundary_conditions('Dirichlet', {102:1, 101:0})
        self.mesh_1.set_boundary_conditions('Neumann', {201:0})

        self.mesh_2 = MeshManager('geometry_two_regions_test.msh', dimension=3)
        self.all_volumes_2 = self.mesh_2.mb.get_entities_by_dimension(0, 3)
        self.mesh_2.set_media_property('Permeability', {1:K_1, 2:K_2})
        self.mesh_2.set_boundary_conditions('Dirichlet', {102:1, 101:0})
        self.mesh_2.set_boundary_conditions('Neumann', {201:0})


    def test_if_method_has_all_dirichlet_nodes(self):
        mpfad = MpfaD3D(self.mesh_2)
        dirichlet_nodes = self.mesh_2.mb.get_entities_by_type_and_tag(
            0, types.MBVERTEX, mesh_2.dirichlet_tag, np.array((None,)))
        self.assertEqual(len(dirichlet_nodes), 10)

    # def test_linear_problem_with_orthogonal_mesh(self):
    #     mpfad = MpfaD3D(self.mesh_1)
    #     mpfad.run()
    #     pressure_tag = self.mesh_1.mb.tag_get_handle('Pressure')
    #     self.
    #     self.assertEqual(len(mpfad.dirichlet_nodes), 10)
