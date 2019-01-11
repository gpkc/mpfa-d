import numpy as np
from math import pi
# from mpfad.debugging_mesh_preprocessor import MpfaD3D
from mpfad.MpfaD import MpfaD3D
from mpfad.interpolation.LPEW3 import LPEW3
from mesh_preprocessor import MeshManager


class ObliqueDrain:

    def __init__(self, filename, delta):
        self.mesh = MeshManager(filename, dim=3)
        self.mesh.set_boundary_condition('Dirichlet', {101: 0.0},
                                         dim_target=2, set_nodes=True)
        self.delta = delta
        self.mesh.set_global_id()
        self.mesh.get_redefine_centre()
        self.mpfad = MpfaD3D(self.mesh)
        self.lpew3 = LPEW3(self.mesh)

    # TODO: print data
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

    def calculate_gradient(self, x, y, R, benchmark, delta=0.00001):
        grad_x = (benchmark(x + delta, y, R)[1] -
                  benchmark(x, y, R)[1]) / delta
        grad_y = (benchmark(x, y + delta, R)[1] -
                  benchmark(x, y, R)[1]) / delta
        grad_z = 0.0
        grad = np.array([grad_x, grad_y, grad_z])
        return grad

    def calculate_K_gradient(self, x, y, R, benchmark):
        perm = np.array(benchmark(x, y, R)[0][0]).reshape([3, 3])
        grad = self.calculate_gradient(x, y, R, benchmark)
        return np.dot(perm, grad)

    def calculate_divergent(self, x, y, R, benchmark, delta=0.00001):
        k_grad_x = (self.calculate_K_gradient(x + delta, y, R, benchmark)[0]
                    - self.calculate_K_gradient(x, y, R, benchmark)[0]) / delta
        k_grad_y = (self.calculate_K_gradient(x, y + delta, R, benchmark)[1]
                    - self.calculate_K_gradient(x, y, R, benchmark)[1]) / delta
        k_grad_z = 0.0
        return -np.sum(k_grad_x + k_grad_y + k_grad_z)

    def _phi1(self, x, y):
        return y - self.delta * (x - .5) - .475

    def _phi2(self, x, y):
        phi1 = self._phi1(x, y)
        return phi1 - .05

    def _rotationMatrix(self):
        theta = np.arctan(self.delta)
        return [np.cos(theta), -np.sin(theta), 0.,
                np.sin(theta), np.cos(theta), 0,
                0, 0, 1]

    def _obliqueDrain(self, x, y, R=None):
        u = -x - self.delta * y
        phi1 = self._phi1(x, y)
        phi2 = self._phi2(x, y)
        # print(phi1, phi2)
        if phi1 <= 0.:
            # print('region 1')
            zone = 1.
            alfa = 1
            beta = .1
            perm = [alfa, 0., 0.,
                    0., beta, 0.,
                    0., 0., 1.]

        if phi1 >= 0. and phi2 <= 0.:
            # print('region 2')
            zone = 2.
            alfa = 100
            beta = 10
            perm = [alfa, 0., 0.,
                    0., beta, 0.,
                    0., 0., 1]

        if phi2 > 0.:
            # print('region 3')
            zone = 3.
            alfa = 1
            beta = .1
            perm = [alfa, 0., 0.,
                    0., beta, 0.,
                    0., 0., 1.]
        try:
            perm = np.dot(np.dot(R, np.array(perm).reshape([3, 3])),
                          np.linalg.inv(R)).reshape([1, 9])
            perm = (R * np.array(perm).reshape([3, 3]) * np.linalg.inv(R)).reshape([1, 9])
        except:
            return None, u, zone
        # perm = [1.0, 0.0, 0.0,
        #         0.0, 1.0, 0.0,
        #         0.0, 0.0, 1.0]
        return perm, u, zone

    def runCase(self, log_name):
        R = np.array(self._rotationMatrix()).reshape([3, 3])
        for node in self.mesh.get_boundary_nodes():
            x, y, z = self.mesh.mb.get_coords([node])
            g_D = self._obliqueDrain(x, y)[1]
            self.mesh.mb.tag_set_data(self.mesh.dirichlet_tag, node, g_D)
        volumes = self.mesh.all_volumes
        vols = []
        for volume in volumes:
            x, y, z = self.mesh.mb.tag_get_data(self.mesh.volume_centre_tag,
                                                volume)[0]
            perm = self._obliqueDrain(x, y, R)[0]
            zone = self._obliqueDrain(x, y, R)[2]
            # print('zone', zone)
            self.mesh.mb.tag_set_data(self.mesh.perm_tag, volume, perm)
            self.mesh.mb.tag_set_data(self.mesh.regions_validation_tag, volume,
                                      zone)
            vol_nodes = self.mesh.mb.get_adjacencies(volume, 0)
            vol_nodes_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_nodes_crds = np.reshape(vol_nodes_crds, (4, 3))
            tetra_vol = self.mesh.get_tetra_volume(vol_nodes_crds)
            vols.append(tetra_vol)
            source_term = 0.
            self.mesh.mb.tag_set_data(self.mesh.source_tag, volume,
                                      source_term * tetra_vol)
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)

        for volume in volumes:
            x, y, z = self.mesh.mb.tag_get_data(self.mesh.volume_centre_tag,
                                                volume)[0]
            analytical_solution = self._obliqueDrain(x, y)[1]
            calculated_solution = self.mpfad.mb.tag_get_data(
                                  self.mpfad.pressure_tag, volume)[0][0]
            print(analytical_solution, calculated_solution,
                  np.abs(analytical_solution - calculated_solution))

        err = []
        u = []
        for volume in volumes:
            x_c, y_c, z_c = self.mesh.mb.tag_get_data(
                            self.mesh.volume_centre_tag, volume)[0]
            analytical_solution = self._obliqueDrain(x_c, y_c)[1]
            calculated_solution = self.mpfad.mb.tag_get_data(
                                  self.mpfad.pressure_tag, volume)[0][0]
            err.append(np.absolute((analytical_solution - calculated_solution))
                       )
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
            f.write('Relat ive L2 norm:\t %.6f\n' % (results[2]))
            f.write('average error:\t %.6f\n' % (results[3]))
            f.write('maximum error:\t %.6f\n' % (results[4]))
            f.write('minimum error:\t %.6f\n' % (results[5]))

        print('erro maximo', max(err))

        print('END OF ' + log_name + '!!!\n')
        self.mpfad.record_data('oblique_drain' + log_name + '.vtk')
