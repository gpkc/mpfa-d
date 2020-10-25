import numpy as np
from math import pi, sqrt
import solvers.helpers.geometric as geo
from solvers.MpfaD import MpfaD3D
from preprocessor.mesh_preprocessor import MeshManager
from pymoab import types


class TestCasesMGE:
    def __init__(self, filename, interpolation_method):
        self.mesh = MeshManager(filename, dim=3)
        self.mesh.set_boundary_condition(
            "Dirichlet", {101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh.set_global_id()
        self.mesh.get_redefine_centre()
        self.mpfad = MpfaD3D(self.mesh)
        self.im = interpolation_method(self.mesh)

    def get_velocity(self, bmk):
        self.node_pressure_tag = self.mpfad.mb.tag_get_handle(
            "Node Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )
        p_verts = []
        for node in self.mesh.all_nodes:
            try:
                p_vert = self.mpfad.mb.tag_get_data(
                    self.mpfad.dirichlet_tag, node
                )
                p_verts.append(p_vert[0])
            except Exception:
                p_vert = 0.0
                p_tag = self.mpfad.pressure_tag
                nd_weights = self.mpfad.nodes_ws[node]
                for volume, wt in nd_weights.items():
                    p_vol = self.mpfad.mb.tag_get_data(p_tag, volume)
                    p_vert += p_vol * wt
                p_verts.append(p_vert[0])
        self.mpfad.mb.tag_set_data(
            self.node_pressure_tag, self.mesh.all_nodes, p_verts
        )
        err = []
        err_grad = []
        grads_p = []
        all_vels = []
        areas = []
        vols = []
        for a_volume in self.mesh.all_volumes:
            vol_faces = self.mesh.mtu.get_bridge_adjacencies(a_volume, 2, 2)
            vol_nodes = self.mesh.mtu.get_bridge_adjacencies(a_volume, 0, 0)
            vol_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_crds = np.reshape(vol_crds, ([4, 3]))
            vol_volume = self.mesh.get_tetra_volume(vol_crds)
            vols.append(vol_volume)
            I, J, K = self.mesh.mtu.get_bridge_adjacencies(vol_faces[0], 2, 0)
            L = list(
                set(vol_nodes).difference(
                    set(
                        self.mesh.mtu.get_bridge_adjacencies(
                            vol_faces[0], 2, 0
                        )
                    )
                )
            )
            JI = self.mesh.mb.get_coords([I]) - self.mesh.mb.get_coords([J])
            JK = self.mesh.mb.get_coords([K]) - self.mesh.mb.get_coords([J])
            LJ = self.mesh.mb.get_coords([J]) - self.mesh.mb.get_coords(L)
            N_IJK = np.cross(JI, JK) / 2.0

            test = np.dot(LJ, N_IJK)
            if test < 0.0:
                I, K = K, I
                JI = self.mesh.mb.get_coords([I]) - self.mesh.mb.get_coords(
                    [J]
                )
                JK = self.mesh.mb.get_coords([K]) - self.mesh.mb.get_coords(
                    [J]
                )
                N_IJK = np.cross(JI, JK) / 2.0

            tan_JI = np.cross(N_IJK, JI)
            tan_JK = np.cross(N_IJK, JK)
            face_area = np.sqrt(np.dot(N_IJK, N_IJK))

            h_L = geo.get_height(N_IJK, LJ)

            p_I = self.mpfad.mb.tag_get_data(self.node_pressure_tag, I)
            p_J = self.mpfad.mb.tag_get_data(self.node_pressure_tag, J)
            p_K = self.mpfad.mb.tag_get_data(self.node_pressure_tag, K)
            p_L = self.mpfad.mb.tag_get_data(self.node_pressure_tag, L)
            grad_normal = -2 * (p_J - p_L) * N_IJK

            grad_cross_I = (p_J - p_I) * (
                (np.dot(tan_JK, LJ) / face_area ** 2) * N_IJK
                - (h_L / (face_area)) * tan_JK
            )
            grad_cross_K = (p_K - p_J) * (
                (np.dot(tan_JI, LJ) / face_area ** 2) * N_IJK
                - (h_L / (face_area)) * tan_JI
            )

            grad_p = -(1 / (6 * vol_volume)) * (
                grad_normal + grad_cross_I + grad_cross_K
            )
            vol_centroid = np.asarray(
                self.mesh.mb.tag_get_data(
                    self.mesh.volume_centre_tag, a_volume
                )[0]
            )
            vol_perm = self.mesh.mb.tag_get_data(
                self.mesh.perm_tag, a_volume
            ).reshape([3, 3])
            x, y, z = vol_centroid
            grad_p_bar = self.calculate_gradient(x, y, z, bmk)
            grads_p.append(np.dot(grad_p_bar, grad_p_bar))
            # print(grad_p, grad_p_bar, grad_p - grad_p_bar)
            e = grad_p[0] - grad_p_bar
            err_norm = np.sqrt(np.dot(e, e))
            # print(err_norm)
            err_grad.append(err_norm ** 2)

            for face in vol_faces:
                # x, y, z = self.mesh.mb.get_coords([face])
                face_nodes = self.mesh.mtu.get_bridge_adjacencies(face, 2, 0)
                face_nodes_crds = self.mesh.mb.get_coords(face_nodes)
                area_vect = geo._area_vector(
                    face_nodes_crds.reshape([3, 3]), vol_centroid
                )[0]
                unit_area_vec = area_vect / np.sqrt(
                    np.dot(area_vect, area_vect)
                )
                k_grad_p = np.dot(vol_perm, grad_p[0])
                vel = -np.dot(k_grad_p, unit_area_vec)
                calc_vel = -np.dot(
                    self.calculate_K_gradient(x, y, z, bmk), unit_area_vec
                )
                err.append(abs(vel - calc_vel) ** 2)
                areas.append(np.sqrt(np.dot(area_vect, area_vect)))
                all_vels.append(calc_vel ** 2)
        norm_vel = np.sqrt(np.dot(err, areas) / np.dot(all_vels, areas))
        # print(len(err_norm), len(vols))
        norm_grad = np.sqrt(np.dot(err_grad, vols) / np.dot(grads_p, vols))
        # print(norm_grad)
        return norm_vel, norm_grad

    def norms_calculator(self, error_vector, volumes_vector, u_vector):
        error_vector = np.array(error_vector)
        volumes_vector = np.array(volumes_vector)
        u_vector = np.array(u_vector)
        l2_norm = np.dot(error_vector, error_vector) ** (1 / 2)
        l2_volume_norm = np.dot(error_vector ** 2, volumes_vector) ** (1 / 2)
        erl2 = (
            np.dot(error_vector ** 2, volumes_vector)
            / np.dot(u_vector ** 2, volumes_vector)
        ) ** (1 / 2)
        avr_error = l2_norm / len(volumes_vector)
        max_error = max(error_vector)
        min_error = min(error_vector)
        results = [
            l2_norm,
            l2_volume_norm,
            erl2,
            avr_error,
            max_error,
            min_error,
        ]
        return results

    def calculate_gradient(self, x, y, z, benchmark, delta=0.00001):
        grad_x = (
            benchmark(x + delta, y, z)[1] - benchmark(x, y, z)[1]
        ) / delta
        grad_y = (
            benchmark(x, y + delta, z)[1] - benchmark(x, y, z)[1]
        ) / delta
        grad_z = (
            benchmark(x, y, z + delta)[1] - benchmark(x, y, z)[1]
        ) / delta
        grad = np.array([grad_x, grad_y, grad_z])
        return grad

    def calculate_K_gradient(self, x, y, z, benchmark):
        perm = np.array(benchmark(x, y, z)[0]).reshape([3, 3])
        grad = self.calculate_gradient(x, y, z, benchmark)
        return np.dot(perm, grad)

    def calculate_divergent(self, x, y, z, benchmark, delta=0.00001):
        k_grad_x = (
            self.calculate_K_gradient(x + delta, y, z, benchmark)[0]
            - self.calculate_K_gradient(x, y, z, benchmark)[0]
        ) / delta
        k_grad_y = (
            self.calculate_K_gradient(x, y + delta, z, benchmark)[1]
            - self.calculate_K_gradient(x, y, z, benchmark)[1]
        ) / delta
        k_grad_z = (
            self.calculate_K_gradient(x, y, z + delta, benchmark)[2]
            - self.calculate_K_gradient(x, y, z, benchmark)[2]
        ) / delta
        return -np.sum(k_grad_x + k_grad_y + k_grad_z)

    def mge_test_case_1(self, x, y, z):
        K = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        u = np.sin(sqrt(2) * pi * x) * np.sinh(pi * y) * np.sinh(pi * z)

        return K, u

    def mge_test_case_2(self, x, y, z):
        if x <= 0.5:
            K = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            u = (x - 0.5) * (2 * np.sin(y) + np.cos(y)) + np.sin(y) + z
        else:
            K = [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0]
            u = np.exp(x) - 0.5 * np.sin(y) + z
        return K, u

    def mge_test_case_3(self, x, y, z):
        eps = 50
        K = [
            (1 + 10 * x ** 2 + y ** 2 + z ** 2),
            0.0,
            0.0,
            0.0,
            (1 + x ** 2 + 10 * y ** 2 + z ** 2),
            0.0,
            0.0,
            0.0,
            eps * (1 + x ** 2 + y ** 2 + 10 * z ** 2),
        ]
        u = x * (1 - x) * y * (1 - y) * z * (1 - z)
        return K, u

    def mge_test_case_4(self, x, y, z):
        eps1 = 0.1
        eps2 = 10
        K = [
            np.cos(pi * x) ** 2 + eps1 * np.sin(pi * x) ** 2,
            np.sin(pi * x) * np.cos(pi * x)
            - eps1 * np.sin(pi * x) * np.cos(pi * x),
            0.0,
            np.sin(pi * x) * np.cos(pi * x)
            - eps1 * np.sin(pi * x) * np.cos(pi * x),
            np.sin(pi * x) ** 2 + eps1 * np.cos(pi * x) ** 2,
            0.0,
            0.0,
            0.0,
            eps2 * (1 + x + y + z),
        ]
        u = np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z)

        return K, u

    def mge_test_case_5(self, x, y, z):
        epsx = 1
        epsy = 1e-3
        epsz = 10
        K = [
            (epsx * x ** 2 + epsy * y ** 2) / (x ** 2 + y ** 2),
            ((epsx - epsy) * x * y) / (x ** 2 + y ** 2),
            0.0,
            ((epsx - epsy) * x * y) / (x ** 2 + y ** 2),
            (epsy * x ** 2 + epsx * y ** 2) / (x ** 2 + y ** 2),
            0.0,
            0.0,
            0.0,
            epsz * (z + 1),
        ]
        u = np.sin(2 * pi * x) * np.sin(2 * pi * y) * np.sin(2 * pi * z)

        return K, u

    def run_case(self, log_name, test_case):
        """
        Call the Finite Volume for Complex Applications Benchmark.

        Test case 2.
        """
        func = {
            "mge_test_case_1": self.mge_test_case_1,
            "mge_test_case_2": self.mge_test_case_2,
            "mge_test_case_3": self.mge_test_case_3,
            "mge_test_case_4": self.mge_test_case_4,
            "mge_test_case_5": self.mge_test_case_5,
        }
        for node in self.mesh.get_boundary_nodes():
            x, y, z = self.mesh.mb.get_coords([node])
            g_D = func[test_case](x, y, z)[1]
            self.mesh.mb.tag_set_data(self.mesh.dirichlet_tag, node, g_D)
        volumes = self.mesh.all_volumes
        vols = []
        for volume in volumes:
            x, y, z = self.mesh.mb.tag_get_data(
                self.mesh.volume_centre_tag, volume
            )[0]
            self.mesh.mb.tag_set_data(
                self.mesh.perm_tag, volume, func[test_case](x, y, z)[0]
            )
            vol_nodes = self.mesh.mb.get_adjacencies(volume, 0)
            vol_nodes_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_nodes_crds = np.reshape(vol_nodes_crds, (4, 3))
            tetra_vol = self.mesh.get_tetra_volume(vol_nodes_crds)
            vols.append(tetra_vol)
            source_term = self.calculate_divergent(x, y, z, func[test_case])
            self.mesh.mb.tag_set_data(
                self.mesh.source_tag, volume, source_term * tetra_vol
            )

        self.mpfad.run_solver(self.im.interpolate)
        err = []
        u = []
        for volume in volumes:
            x, y, z = self.mesh.mb.tag_get_data(
                self.mesh.volume_centre_tag, volume
            )[0]
            analytical_solution = func[test_case](x, y, z)[1]
            calculated_solution = self.mpfad.mb.tag_get_data(
                self.mpfad.pressure_tag, volume
            )[0][0]
            err.append(
                np.absolute((analytical_solution - calculated_solution))
            )
            u.append(analytical_solution)
        u_max = max(
            self.mpfad.mb.tag_get_data(self.mpfad.pressure_tag, volumes)
        )
        u_min = min(
            self.mpfad.mb.tag_get_data(self.mpfad.pressure_tag, volumes)
        )
        results = self.norms_calculator(err, vols, u)
        non_zero_mat = self.mpfad.T.NumGlobalNonzeros()
        norm_vel, norm_grad = self.get_velocity(func[test_case])
        path = (
            f"paper_mpfad_tests/mge_paper_cases/{func[test_case].__name__}/"
            + log_name
            + "_log"
        )
        with open(path, "w") as f:
            f.write("TEST CASE 2\n\nUnknowns:\t %.6f\n" % (len(volumes)))
            f.write("Non-zero matrix:\t %.6f\n" % (non_zero_mat))
            f.write("Umin:\t %.6f\n" % (u_min))
            f.write("Umax:\t %.6f\n" % (u_max))
            f.write("L2 norm:\t %.6f\n" % (results[0]))
            f.write("l2 norm volume weighted:\t %.6f\n" % (results[1]))
            f.write("Relative L2 norm:\t %.6f\n" % (results[2]))
            f.write("average error:\t %.6f\n" % (results[3]))
            f.write("maximum error:\t %.6f\n" % (results[4]))
            f.write("minimum error:\t %.6f\n" % (results[5]))
            f.write("velocity norm: \t %.6g\n" % norm_vel)
            f.write("gradient norm: \t %.6g\n" % norm_grad)
        print("max error: ", max(err), "l-2 relative norm: ", results[2])
        path = (
            f"paper_mpfad_tests/mge_paper_cases/{func[test_case].__name__}/"
            + log_name
        )
        self.mpfad.record_data(path + ".vtk")
        print("END OF " + log_name + "!!!\n")
