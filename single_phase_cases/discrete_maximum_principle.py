import numpy as np
from solvers.nMpfaD import MpfaD3D
from preprocessor.mesh_preprocessor import MeshManager
from solvers.interpolation.LSW import LSW


class DiscreteMaxPrinciple:
    def __init__(self, filename, interpolation_method):
        self.mesh = MeshManager(filename, dim=3)
        self.mesh.set_global_id()
        self.mesh.get_redefine_centre()
        self.mx = 2.0
        self.mn = 0.0
        self.mesh.set_boundary_condition(
            "Dirichlet",
            {51: self.mx, 10: self.mn},
            dim_target=2,
            set_nodes=True,
        )
        self.mpfad = MpfaD3D(self.mesh)
        self.im = interpolation_method(self.mesh)
        self.precond = LSW(self.mesh)

    def rotation_matrix(self, theta, axis=0):
        """
        Return the rotation matrix.

        Arguments:
        double theta = angle (in radians) for the rotation
        int axis = axis for rotation

        retuns a numpy array with shape [3, 3]
        """
        R_matrix = np.zeros([3, 3])
        cos = np.cos(theta)
        sin = np.sin(theta)

        if axis == 0:
            R_matrix = np.array([1, 0, 0, 0, cos, -sin, 0, sin, cos])
        if axis == 1:
            R_matrix = np.array([cos, 0, sin, 0, 1, 0, -sin, 0, cos])
        else:
            R_matrix = np.array([cos, -sin, 0, sin, cos, 0, 0, 0, 1])

        R_matrix = R_matrix.reshape([3, 3])
        return R_matrix

    def get_perm_tensor(self):
        R_x = self.rotation_matrix(-np.pi / 3, axis=0)
        R_y = self.rotation_matrix(-np.pi / 4, axis=1)
        R_z = self.rotation_matrix(-np.pi / 6, axis=2)
        R_xyz = np.matmul(np.matmul(R_z, R_y), R_x)
        perm = np.diag([300, 15, 1])
        perm = np.matmul(np.matmul(R_xyz, perm), R_xyz.transpose()).reshape(
            [1, 9]
        )[0]

        perms = []
        for volume in self.mesh.all_volumes:
            perms.append(perm)
        self.mesh.mb.tag_set_data(
            self.mesh.perm_tag, self.mesh.all_volumes, perms
        )

    def run_dmp(self, log_name):
        self.get_perm_tensor()
        # precond = self.mpfad.run_solver(self.precond.interpolate)
        # x = self.mpfad.x
        self.mpfad = MpfaD3D(self.mesh)
        self.mpfad.run_solver(self.im.interpolate)
        max_p = max(self.mpfad.x)
        min_p = min(self.mpfad.x)
        volumes = self.mesh.all_volumes
        oversh = []
        undersh = []
        for volume in volumes:
            pressure = self.mesh.mb.tag_get_data(
                self.mesh.pressure_tag, volume
            )[0][0]
            x, y, z = self.mesh.mb.tag_get_data(
                self.mesh.volume_centre_tag, volume
            )[0]
            vol_nodes = self.mesh.mb.get_adjacencies(volume, 0)
            vol_nodes_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_nodes_crds = np.reshape(vol_nodes_crds, (4, 3))
            tetra_vol = self.mesh.get_tetra_volume(vol_nodes_crds)
            eps_mx = max(pressure - self.mx, 0.0) ** 2
            oversh.append(eps_mx * tetra_vol)
            eps_mn = min(pressure - self.mn, 0.0) ** 2
            undersh.append(eps_mn * tetra_vol)
        overshooting = sum(oversh) ** (1 / 2.0)
        undershooting = sum(undersh) ** (1 / 2.0)
        print(
            max_p, min_p, sum(oversh) ** (1 / 2.0), sum(undersh) ** (1 / 2.0)
        )

        path = "paper_mpfad_tests/dmp_tests/" + log_name
        with open(path + "_log", "w") as f:
            f.write("\nUnknowns:\t %.0f\n" % (len(self.mesh.all_volumes)))
            f.write("Umin:\t %.6f\n" % (min_p))
            f.write("Umax:\t %.6f\n" % (max_p))
            f.write("Overshooting:\t %.6f\n" % (overshooting))
            f.write("Undershooting:\t %.6f\n" % (undershooting))
            f.write(
                "Non-zero matrix:\t %.0f\n"
                % (self.mpfad.T.NumGlobalNonzeros())
            )
        self.mpfad.record_data(path + ".vtk")

    def perm_tensor_lai(self, x, y, z):
        e = 5e-3

        k = np.asarray(
            [
                y ** 2 + e * x ** 2,
                -(1 - e) * x * y,
                0,
                -(1 - e) * x * y,
                x ** 2 + e * y ** 2,
                0,
                0.0,
                0.0,
                1.0,
            ]
        )
        return k

    def run_lai_sheng_dmp_test(self):
        all_volumes = self.mesh.all_volumes
        for volume in all_volumes:
            x, y, z = self.mesh.mb.get_coords(volume)
            perm = self.perm_tensor_lai(x, y, z)
            self.mesh.mb.tag_set_data(self.mesh.perm_tag, volume, perm)
            if x < 5 / 8 and x > 3 / 8 and y < 5 / 8 and y > 3 / 8:
                source_term = 1.0
                self.mesh.mb.tag_set_data(
                    self.mesh.source_tag, volume, source_term
                )
            else:
                source_term = 0.0
                self.mesh.mb.tag_set_data(
                    self.mesh.source_tag, volume, source_term
                )
        self.mpfad.run_solver(self.im.interpolate)
        solution = self.mesh.mb.tag_get_data(
            self.mesh.pressure_tag, all_volumes
        )
        print("min: ", min(solution), "max: ", max(solution))
