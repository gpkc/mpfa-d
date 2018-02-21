# import unittest
# from MPFAD_3D.pressure_solver_3D import MpfaD3D
# from MPFAD_3D.mesh_processor import MeshManager
# from unittest import mock
#
#
# class PressureSolverTest(unittest.TestCase):
#
#     def setUp(self):
#         K_1 = np.array([2.0, 0.0, 0.0,
#                         0.0, 2.0, 0.0,
#                         0.0, 0.0, 2.0])
#         self.mesh_1 = MeshManager('geometry_3D_test.msh')
#         self.mesh_1.set_media_property({1:K_1})
#         self.mesh_1.set_boundary_conditions()
#
#
#     def test_linear_problem_with_orthogonal_mesh(self):
#         mpfad = MpfaD3D(self.mesh_1)
