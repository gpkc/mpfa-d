import unittest
import numpy as np
from math import pi
from mpfad.MpfaD import MpfaD3D
from mpfad.interpolation.IDW import IDW
from mpfad.interpolation.LSW import LSW
from mpfad.interpolation.LPEW3 import LPEW3
from mesh_preprocessor import MeshManager


class InterpMethodTest(unittest.TestCase):

    def setUp(self):
        K = [1.0, 0.5, 0.0,
             0.5, 1.0, 0.5,
             0.0, 0.5, 1.0]
        self.mesh = MeshManager('mesh_test_4.msh', dim=3)
        self.mesh.set_media_property('Permeability', {1: K}, dim_target=3)
        self.mesh.set_boundary_condition('Dirichlet', {101: 1.0},
                                         dim_target=2, set_nodes=True)
        self.mpfad = MpfaD3D(self.mesh)

    def benchmark_1_solution(self, x, y, z):
        g_D = 1 + np.sin(pi * x) * np.sin(pi * (y + 1/2)) * np.sin(pi * (z +
                                                                         1/3))

        return g_D

    def test_benchmark_case_1(self):
        boundary_nodes = self.mesh.get_boundary_nodes()
        for node in boundary_nodes:
            x, y, z = self.mesh.mb.get_coords([node])
            g_D = self.benchmark_1_solution(x, y, z)
            self.mesh.mb.tag_set_data(self.mesh.dirichlet_tag, node, g_D)
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        volumes = self.mesh.all_volumes
        print(volumes)
        """
        for volume in volumes:
            x_c, y_c, z_c = self.mesh.get_centroid(volume)
            analytical_solution = self.benchmark_1_solution(x_c, y_c, z_c)
            calculated_solution = self.mpfad.mb.tag_get_data(
                                  self.mpfad.pressure_tag, volume)[0][0]
            err = abs(analytical_solution -
                      calculated_solution) / analytical_solution
            print(err)
        """
