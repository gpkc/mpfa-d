import numpy as np
from math import pi
# import mpfad.helpers.geometric as geo
from mpfad.MpfaD import MpfaD3D
from mesh_preprocessor import MeshManager
from pymoab import types


class BenchmarkFVCA:

    def __init__(self, filename, interpolation_method):
        self.mesh = MeshManager(filename, dim=3)
        self.mesh.set_boundary_condition('Dirichlet', {101: None},
                                         dim_target=2, set_nodes=True)
        self.mesh.set_global_id()
        self.mesh.get_redefine_centre()
        self.mpfad = MpfaD3D(self.mesh)
        self.im = interpolation_method(self.mesh)

    # def get_node_pressure(self, node):
    #     try:
    #         p_vert = self.mpfad.mb.tag_get_data(self.mpfad.dirichlet_tag, node)
    #     except:
    #         p_vert = 0.0
    #         p_tag = self.mpfad.pressure_tag
    #         nd_weights = self.mpfad.nodes_ws[node]
    #         for volume, wt in nd_weights.items():
    #             p_vol = self.mpfad.mb.tag_get_data(p_tag, volume)
    #             p_vert += p_vol * wt
    #     return p_vert
    #
    # def get_velocity(self, face):
    #     if face in self.mpfad.dirichlet_faces:
    #         D_JK, D_JI, K_eq, I, J, K, face_area \
    #          = self.mpfad.mb.tag_get_data(self.mpfad.flux_info_tag, face)[0]
    #         adj_vol = self.mesh.mtu.get_bridge_adjacencies(face, 2, 3)
    #         p_L = self.mesh.mb.tag_get_data(self.mesh.pressure_tag, adj_vol)
    #         p_I = self.get_node_pressure(int(I))
    #         p_J = self.get_node_pressure(int(J))
    #         p_K = self.get_node_pressure(int(K))
    #         v = - (D_JK * (p_I - p_J)
    #                - K_eq * (p_J - p_L)
    #                + D_JI * (p_J - p_K)) * face_area
    #         # TODO: Implementation of analitical velocity calculation
    #     else:
    #         return None
        # if face in self.mpfad.intern_faces:
        #     D_JK, D_JI, K_eq, I, J, K, face_area \
        #      = self.mpfad.mb.tag_get_data(self.mpfad.flux_info_tag, face)[0]
        #     left_volume, right_volume = \
        #         self.mesh.mtu.get_bridge_adjacencies(face, 2, 3)
        #     L = self.mesh.mb.tag_get_data(self.mesh.volume_centre_tag,
        #                                   left_volume)[0]
        #     R = self.mesh.mb.tag_get_data(self.mesh.volume_centre_tag,
        #                                   right_volume)[0]
        #     dist_LR = R - L
        #     I, J, K = self.mesh.mtu.get_bridge_adjacencies(face, 0, 0)
        #     JI = self.mesh.mb.get_coords([I]) - self.mesh.mb.get_coords([J])
        #     JK = self.mesh.mb.get_coords([K]) - self.mesh.mb.get_coords([J])
        #
        #     N_IJK = np.cross(JI, JK) / 2.
        #     test = np.dot(N_IJK, dist_LR)
        #
        #     if test < 0:
        #         left_volume, right_volume = right_volume, left_volume
        #
        #     p_I = self.get_node_pressure(int(I))
        #     p_J = self.get_node_pressure(int(J))
        #     p_K = self.get_node_pressure(int(K))
        #     p_R, p_L = self.mesh.mb.tag_get_data(self.mesh.pressure_tag,
        #                                          [left_volume, right_volume])
        #     v = -K_eq * face_area * (2 * (p_R - p_L)
        #                              - D_JK * (p_J - p_I)
        #                              + D_JI * (p_J - p_K))
        # return v

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

    def benchmark_case_1(self, log_name):
        for node in self.mesh.get_boundary_nodes():
            x, y, z = self.mesh.mb.get_coords([node])
            g_D = self._benchmark_1(x, y, z)[1]
            self.mesh.mb.tag_set_data(self.mesh.dirichlet_tag, node, g_D)
        volumes = self.mesh.all_volumes
        faces = self.mesh.all_faces
        vols = []
        for volume in volumes:
            x, y, z = self.mesh.mb.tag_get_data(self.mesh.volume_centre_tag,
                                                volume)[0]
            self.mesh.mb.tag_set_data(self.mesh.perm_tag, volume,
                                      self._benchmark_1(x, y, z)[0])
            vol_nodes = self.mesh.mb.get_adjacencies(volume, 0)
            vol_nodes_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_nodes_crds = np.reshape(vol_nodes_crds, (4, 3))
            tetra_vol = self.mesh.get_tetra_volume(vol_nodes_crds)
            vols.append(tetra_vol)
            source_term = self.calculate_divergent(x, y, z, self._benchmark_1)
            self.mesh.mb.tag_set_data(self.mesh.source_tag, volume,
                                      source_term * tetra_vol)

        self.mpfad.run_solver(self.im.interpolate)
        u_err = []
        u = []
        for volume in volumes:
            x, y, z = self.mesh.mb.tag_get_data(self.mesh.volume_centre_tag,
                                                volume)[0]
            analytical_solution = self._benchmark_1(x, y, z)[1]
            calculated_solution = self.mpfad.mb.tag_get_data(
                                  self.mpfad.pressure_tag, volume)[0][0]
            u_err.append(np.absolute((analytical_solution
                                     - calculated_solution)))
            u.append(analytical_solution)
        u_max = max(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        u_min = min(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        results = self.norms_calculator(u_err, vols, u)
        non_zero_mat = self.mpfad.T.NumGlobalNonzeros()
        path = 'paper_mpfad_tests/benchmark_fvca_cases/benchmark_case_1/' \
            + log_name + '_log'
        with open(path, 'w') as f:
            f.write('TEST CASE 1\n\nUnknowns:\t %.0f\n' % (len(volumes)))
            f.write('Non-zero matrix:\t %.0f\n' % (non_zero_mat))
            f.write('Umin:\t %.6f\n' % (u_min))
            f.write('Umax:\t %.6f\n' % (u_max))
            f.write('L2 norm:\t %.6g\n' % (results[0]))
            f.write('l2 norm volume weighted:\t %.6g\n' % (results[1]))
            f.write('Relative L2 norm:\t %.6g\n' % (results[2]))
            f.write('average error:\t %.6g\n' % (results[3]))
            f.write('maximum error:\t %.6g\n' % (results[4]))
            f.write('minimum error:\t %.6g\n' % (results[5]))

        # for face in faces:
        #     v = self.get_velocity(face)
        #     print(v)

        print('max error: ', max(u_err), 'l-2 relative norm: ', results[2])
        path = 'paper_mpfad_tests/benchmark_fvca_cases/benchmark_case_1/'
        self.mpfad.record_data(path + log_name + '.vtk')
        print('END OF ' + log_name + '!!!\n')

    def benchmark_case_2(self, log_name):
        for node in self.mesh.get_boundary_nodes():
            x, y, z = self.mesh.mb.get_coords([node])
            g_D = self._benchmark_2(x, y, z)[1]
            self.mesh.mb.tag_set_data(self.mesh.dirichlet_tag, node, g_D)
        volumes = self.mesh.all_volumes
        vols = []
        for volume in volumes:
            x, y, z = self.mesh.mb.tag_get_data(self.mesh.volume_centre_tag,
                                                volume)[0]
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
        self.mpfad.run_solver(self.im.interpolate)
        err = []
        u = []
        for volume in volumes:
            x, y, z = self.mesh.mb.tag_get_data(self.mesh.volume_centre_tag,
                                                volume)[0]
            analytical_solution = self._benchmark_2(x, y, z)[1]
            calculated_solution = self.mpfad.mb.tag_get_data(
                                  self.mpfad.pressure_tag, volume)[0][0]
            err.append(np.absolute((analytical_solution
                                    - calculated_solution)))
            u.append(analytical_solution)
        u_max = max(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        u_min = min(self.mpfad.mb.tag_get_data(
                              self.mpfad.pressure_tag, volumes))
        results = self.norms_calculator(err, vols, u)
        non_zero_mat = self.mpfad.T.NumGlobalNonzeros()
        path = 'paper_mpfad_tests/benchmark_fvca_cases/benchmark_case_2/' \
            + log_name + '_log'
        with open(path, 'w') as f:
            f.write('TEST CASE 2\n\nUnknowns:\t %.6f\n' % (len(volumes)))
            f.write('Non-zero matrix:\t %.6f\n' % (non_zero_mat))
            f.write('Umin:\t %.6f\n' % (u_min))
            f.write('Umax:\t %.6f\n' % (u_max))
            f.write('L2 norm:\t %.6f\n' % (results[0]))
            f.write('l2 norm volume weighted:\t %.6f\n' % (results[1]))
            f.write('Relative L2 norm:\t %.6f\n' % (results[2]))
            f.write('average error:\t %.6f\n' % (results[3]))
            f.write('maximum error:\t %.6f\n' % (results[4]))
            f.write('minimum error:\t %.6f\n' % (results[5]))

        print('max error: ', max(err), 'l-2 relative norm: ', results[2])
        path = 'paper_mpfad_tests/benchmark_fvca_cases/benchmark_case_2/'
        self.mpfad.record_data(path + log_name + '.vtk')
        print('END OF ' + log_name + '!!!\n')
