import unittest
import numpy as np
from math import pi
from mpfad.MpfaD import MpfaD3D
from mpfad.interpolation.LPEW3 import LPEW3
from mesh_preprocessor import MeshManager


class InterpMethodTest(unittest.TestCase):

    def setUp(self):
        K = self.benchmark_1(0., 0., 0.)[0]
        self.mesh = MeshManager('benchmarkmesh.msh', dim=3)
        self.mesh.set_media_property('Permeability', {1: K}, dim_target=3)
        self.mesh.set_boundary_condition('Dirichlet', {101: 0.0},
                                         dim_target=2, set_nodes=True)
        self.mpfad = MpfaD3D(self.mesh)
        self.lpew3 = LPEW3(self.mesh)

    def benchmark_1(self, x, y, z):
        K = [1.0, 0.5, 0.0,
             0.5, 1.0, 0.5,
             0.0, 0.5, 1.0]
        u1 = 1 + np.sin(pi * x) * np.sin(pi * (y + 1/2)) * np.sin(pi * (z +
                                                                        1/3))
        return K, u1

    def benchmark_2(self, x, y, z):
        k_11 = y ** 2 + z ** 2 + 1
        k_12 = - x * y
        k_13 = - x * z
        k_21 = - x * y
        k_22 = x ** 2 + z ** 2 + 1
        k_23 = - y * z
        k_31 = - x * z
        k_32 = - y * z
        k_33 = x ** 2 + y ** 2 + 1

        K = [k_11, k_12, k_13,
             k_21, k_22, k_23,
             k_31, k_32, k_33]

        u2 = (x ** 3 * y ** 2 * z) + x * np.sin(2 * pi * x *
                                                z) * np.sin(2 * pi * x *
                                                            y) * np.sin(2 *
                                                                        pi * z)

        return K, u2

    def benchmark_3(self, x, y, z):
        K = [1E-0, 0E-0, 0E-0,
             0E-0, 1E-0, 0E-0,
             0E-0, 0E-0, 1E-3]
        u3 = np.sin(2 * pi * x) * np.sin(2 * pi * y) * np.sin(2 * pi * z)

        return K, u3

    # @unittest.skip("later")
    def test_benchmark_case_1(self):
        boundary_nodes = self.mesh.get_boundary_nodes()
        for node in boundary_nodes:
            x, y, z = self.mesh.mb.get_coords([node])
            g_D = self.benchmark_1(x, y, z)[1]
            self.mesh.mb.tag_set_data(self.mesh.dirichlet_tag, node, g_D)
        volumes = self.mesh.all_volumes
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        rel2 = []
        print(len(volumes))
        for volume in volumes:
            x_c, y_c, z_c = self.mesh.get_centroid(volume)
            analytical_solution = self.benchmark_1(x_c, y_c, z_c)[1]
            calculated_solution = self.mpfad.mb.tag_get_data(
                                  self.mpfad.pressure_tag, volume)[0][0]
            err = abs(analytical_solution -
                      calculated_solution) / analytical_solution
            rel2.append(err)
        max_p = max(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        min_p = min(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        print('maximo e minimo', max_p, min_p, sum(rel2) / len(rel2))
