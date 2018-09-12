import numpy as np
from math import pi
# from mpfad.debugging_mesh_preprocessor import MpfaD3D
from mpfad.MpfaD import MpfaD3D
from mpfad.interpolation.LPEW3 import LPEW3
from mesh_preprocessor import MeshManager


class BenchmarkFVCA:

    def __init__(self, filename):
        self.mesh = MeshManager(filename, dim=3)
        self.mesh.set_boundary_condition('Dirichlet', {101: 0.0},
                                         dim_target=2, set_nodes=True)
        self.mpfad = MpfaD3D(self.mesh)
        self.lpew3 = LPEW3(self.mesh)

    def record_data(self):
        pass

    def norms_calculator(self, error_vector, volumes_vector, u_vector):
        error_vector = np.array(error_vector)
        volumes_vector = np.array(volumes_vector)
        u_vector = np.array(u_vector)
        l2_norm = np.dot(error_vector, error_vector) ** (1 / 2)
        l2_volume_norm = np.dot(error_vector ** 2, volumes_vector) ** (1 / 2)
        erl2 = (np.dot(error_vector ** 2, volumes_vector) /
                np.dot(u_vector ** 2, volumes_vector)) ** (1 / 2)
        avr_error = l2_norm / len(volumes_vector)
        max_error = max(error_vector)
        min_error = min(error_vector)
        results = [l2_norm, l2_volume_norm, erl2,
                   avr_error, max_error, min_error]
        return results

    def calculate_gradient(self, x, y, z, benchmark, delta=0.00001):
        grad_x = (benchmark(x + delta, y, z)[1] -
                  benchmark(x, y, z)[1]) / delta
        grad_y = (benchmark(x, y + delta, z)[1] -
                  benchmark(x, y, z)[1]) / delta
        grad_z = (benchmark(x, y, z + delta)[1] -
                  benchmark(x, y, z)[1]) / delta
        grad = np.array([grad_x, grad_y, grad_z])
        return grad

    def calculate_K_gradient(self, x, y, z, benchmark):
        perm = np.array(benchmark(x, y, z)[0]).reshape([3, 3])
        grad = self.calculate_gradient(x, y, z, benchmark)
        return np.dot(perm, grad)

    def calculate_divergent(self, x, y, z, benchmark, delta=0.00001):
        k_grad_x = (self.calculate_K_gradient(x + delta, y, z, benchmark)[0]
                    - self.calculate_K_gradient(x, y, z, benchmark)[0]) / delta
        k_grad_y = (self.calculate_K_gradient(x, y + delta, z, benchmark)[1]
                    - self.calculate_K_gradient(x, y, z, benchmark)[1]) / delta
        k_grad_z = (self.calculate_K_gradient(x, y, z + delta, benchmark)[2]
                    - self.calculate_K_gradient(x, y, z, benchmark)[2]) / delta
        return -np.sum(k_grad_x + k_grad_y + k_grad_z)

    def _benchmark_1(self, x, y, z):
        K = [1.0, 0.5, 0.0,
             0.5, 1.0, 0.5,
             0.0, 0.5, 1.0]
        y = y + 1/2.
        z = z + 1/3.
        u1 = 1 + np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z)
        return K, u1

    def _benchmark_2(self, x, y, z):
        k_xx = y ** 2 + z ** 2 + 1
        k_xy = - x * y
        k_xz = - x * z
        k_yx = - x * y
        k_yy = x ** 2 + z ** 2 + 1
        k_yz = - y * z
        k_zx = - x * z
        k_zy = - y * z
        k_zz = x ** 2 + y ** 2 + 1

        K = [k_xx, k_xy, k_xz,
             k_yx, k_yy, k_yz,
             k_zx, k_zy, k_zz]
        # K = [x + 1, 0.0, 0.0,
        #      0.0, y + 1, 0.0,
        #      0.0, 0.0, z + 1]
        u2 = ((x ** 3 * y ** 2 * z) + x * np.sin(2 * pi * x * z)
                                        * np.sin(2 * pi * x * y)
                                        * np.sin(2 * pi * z))

        return K, u2

    def _benchmark_3(self, x, y, z):
        K = [1E-0, 0E-0, 0E-0,
             0E-0, 1E-0, 0E-0,
             0E-0, 0E-0, 1E-3]
        u3 = np.sin(2 * pi * x) * np.sin(2 * pi * y) * np.sin(2 * pi * z)

        return K, u3

    def memory_test(self, log_name):
        info_set = ['iterating over verts',
                    'iterating over edges',
                    'iterating over faces',
                    'iterating over volumes']
        for info, dim in zip(info_set, range(0, 4)):
            print(info)
            entities = self.mesh.mb.get_entities_by_dimension(
                self.mesh.root_set, dim)
            for entity in entities:
                for d in range(2, 3):
                    stuff = self.mesh.mtu.get_bridge_adjacencies(entity, dim, d)
                    print(stuff)


        # for elem in in self.mb.get_entities_by_dimension:
        #     all_faces = self.mesh.mtu.get_bridge_adjacencies(vert, 0, 2)
        #     all_volumes = self.mesh.mtu.get_bridge_adjacencies(vert, 0, 3)

    def benchmark_case_1(self, log_name):
        for node in self.mesh.get_boundary_nodes():
            x, y, z = self.mesh.mb.get_coords([node])
            g_D = self._benchmark_1(x, y, z)[1]
            self.mesh.mb.tag_set_data(self.mesh.dirichlet_tag, node, g_D)
        volumes = self.mesh.all_volumes
        vols = []
        for volume in volumes:
            x, y, z = self.mesh.get_centroid(volume)
            self.mesh.mb.tag_set_data(self.mesh.perm_tag, volume,
                                      self._benchmark_1(x, y, z)[0])
            vol_nodes = self.mesh.mb.get_adjacencies(volume, 0)
            vol_nodes_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_nodes_crds = np.reshape(vol_nodes_crds, (4, 3))
            tetra_vol = self.mesh.get_tetra_volume(vol_nodes_crds)
            vols.append(tetra_vol)
            source_term = self.calculate_divergent(x, y, z,  self._benchmark_1)
            self.mesh.mb.tag_set_data(self.mesh.source_tag, volume,
                                      source_term * tetra_vol)
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        err = []
        u = []
        for volume in volumes:
            x_c, y_c, z_c = self.mesh.get_centroid(volume)
            analytical_solution = self._benchmark_1(x_c, y_c, z_c)[1]
            calculated_solution = self.mpfad.mb.tag_get_data(
                                  self.mpfad.pressure_tag, volume)[0][0]
            err.append(np.absolute((analytical_solution - calculated_solution)))
            u.append(analytical_solution)
        u_max = max(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        u_min = min(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        results = self.norms_calculator(err, vols, u)
        non_zero_mat = np.nonzero(self.mpfad.A_prime)[0]
        with open(log_name, 'w') as f:
            f.write('TEST CASE 1\n\nUnknowns:\t %.6f\n' % (len(volumes)))
            f.write('Non-zero matrix:\t %.6f\n' % (len(non_zero_mat)))
            f.write('Umin:\t %.6f\n' % (u_min))
            f.write('Umax:\t %.6f\n' % (u_max))
            f.write('L2 norm:\t %.6f\n' % (results[0]))
            f.write('l2 norm volume weighted:\t %.6f\n' % (results[1]))
            f.write('Relative L2 norm:\t %.6f\n' % (results[2]))
            f.write('average error:\t %.6f\n' % (results[3]))
            f.write('maximum error:\t %.6f\n' % (results[4]))
            f.write('minimum error:\t %.6f\n' % (results[5]))

        print('END OF ' + log_name + '!!!\n')
        self.mpfad.record_data('benchmark_1' + log_name + '.vtk')

    def benchmark_case_2(self, log_name):
        for node in self.mesh.get_boundary_nodes():
            x, y, z = self.mesh.mb.get_coords([node])
            g_D = self._benchmark_2(x, y, z)[1]
            self.mesh.mb.tag_set_data(self.mesh.dirichlet_tag, node, g_D)
        volumes = self.mesh.all_volumes
        vols = []
        for volume in volumes:
            x, y, z = self.mesh.get_centroid(volume)
            self.mesh.mb.tag_set_data(self.mesh.perm_tag, volume,
                                      self._benchmark_2(x, y, z)[0])
            vol_nodes = self.mesh.mb.get_adjacencies(volume, 0)
            vol_nodes_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_nodes_crds = np.reshape(vol_nodes_crds, (4, 3))
            tetra_vol = self.mesh.get_tetra_volume(vol_nodes_crds)
            vols.append(tetra_vol)
            source_term = self.calculate_divergent(x, y, z,  self._benchmark_2)
            self.mesh.mb.tag_set_data(self.mesh.source_tag, volume,
                                      source_term * tetra_vol)
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        err = []
        u = []
        for volume in volumes:
            x_c, y_c, z_c = self.mesh.get_centroid(volume)
            analytical_solution = self._benchmark_2(x_c, y_c, z_c)[1]
            calculated_solution = self.mpfad.mb.tag_get_data(
                                  self.mpfad.pressure_tag, volume)[0][0]
            err.append(np.absolute((analytical_solution - calculated_solution)))
            u.append(analytical_solution)
        u_max = max(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        u_min = min(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        results = self.norms_calculator(err, vols, u)
        non_zero_mat = np.nonzero(self.mpfad.A)[0]
        with open(log_name, 'w') as f:
            f.write('TEST CASE 1\n\nUnknowns:\t %.6f\n' % (len(volumes)))
            f.write('Non-zero matrix:\t %.6f\n' % (len(non_zero_mat)))
            f.write('Umin:\t %.6f\n' % (u_min))
            f.write('Umax:\t %.6f\n' % (u_max))
            f.write('L2 norm:\t %.6f\n' % (results[0]))
            f.write('l2 norm volume weighted:\t %.6f\n' % (results[1]))
            f.write('Relative L2 norm:\t %.6f\n' % (results[2]))
            f.write('average error:\t %.6f\n' % (results[3]))
            f.write('maximum error:\t %.6f\n' % (results[4]))
            f.write('minimum error:\t %.6f\n' % (results[5]))

        print('END OF ' + log_name + '!!!\n')

    def benchmark_case_3(self, log_name):
        for node in self.mesh.get_boundary_nodes():
            x, y, z = self.mesh.mb.get_coords([node])
            g_D = self._benchmark_3(x, y, z)[1]
            self.mesh.mb.tag_set_data(self.mesh.dirichlet_tag, node, g_D)
        volumes = self.mesh.all_volumes
        for volume in volumes:
            self.mesh.mb.tag_set_data(self.mesh.perm_tag, volume,
                                      self._benchmark_3(0., 0., 0.,)[0])
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        err = []
        vols = []
        u = []
        for volume in volumes:
            x_c, y_c, z_c = self.mesh.get_centroid(volume)
            analytical_solution = self._benchmark_3(x_c, y_c, z_c)[1]
            calculated_solution = self.mpfad.mb.tag_get_data(
                                  self.mpfad.pressure_tag, volume)[0][0]
            tetra_nodes = self.mpfad.mtu.get_bridge_adjacencies(volume, 3, 0)
            tetra_coords = self.mpfad.mb.get_coords(tetra_nodes).reshape([4, 3]
                                                                         )
            tetra_vol = self.mesh.get_tetra_volume(tetra_coords)
            err.append(np.absolute((analytical_solution - calculated_solution)))
            vols.append(tetra_vol)
            u.append(analytical_solution)
        u_max = max(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        u_min = min(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        results = self.norms_calculator(err, vols, u)
        non_zero_mat = np.nonzero(self.mpfad.A)[0]
        with open(log_name, 'w') as f:
            # l2_norm, l2_volume_norm, erl2,
            #            avr_error, max_error, min_error
            f.write('TEST CASE 1\n\nUnknowns:\t %.6f\n' % (len(volumes)))
            f.write('Non-zero matrix:\t %.6f\n' % (len(non_zero_mat)))
            f.write('Umin:\t %.6f\n' % (u_min))
            f.write('Umax:\t %.6f\n' % (u_max))
            f.write('L2 norm:\t %.6f\n' % (results[0]))
            f.write('l2 norm volume weighted:\t %.6f\n' % (results[1]))
            f.write('Relative L2 norm:\t %.6f\n' % (results[2]))
            f.write('average error:\t %.6f\n' % (results[3]))
            f.write('maximum error:\t %.6f\n' % (results[4]))
            f.write('minimum error:\t %.6f\n' % (results[5]))

        print('END OF ' + log_name + '!!!\n')
        self.mpfad.record_data('benchmark_1' + log_name + '.vtk')


            # f.write('Relative L2 norm:\t %.4f\n' % (rl2_norm))
        # print("Test case 2", len(volumes), len(non_zero_mat), u_max, u_min, l2_norm)
