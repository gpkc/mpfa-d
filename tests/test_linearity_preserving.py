import unittest
import numpy as np
from mpfad.MpfaD import MpfaD3D
# from mpfad.interpolation.IDW import IDW
# from mpfad.interpolation.LSW import LSW
from mpfad.interpolation.LPEW3 import LPEW3
# from mpfad.interpolation.LPEW2 import LPEW2
from mesh_preprocessor import MeshManager


class LinearityPreservingTests(unittest.TestCase):

    def setUp(self):

        self.K_1 = np.array([2.0, 1.0, 0.0,
                             1.0, 2.0, 1.0,
                             0.0, 1.0, 2.0])
        self.K_2 = np.array([10.0, 1.0, 0.0,
                             1.0, 10.0, 1.0,
                             0.0, 1.0, 10.0])
        self.K_3 = np.array([1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1E-3])
        self.K_4 = np.array([1.0, 0.0, 0.0,
                             0.0, 1E3, 0.0,
                             0.0, 0.0, 1.0])

        self.mesh_homogeneous = MeshManager('test_mesh_5_vols.h5m', dim=3)
        self.mesh_homogeneous.set_boundary_condition('Dirichlet',
                                                     {101: 0.0},
                                                     dim_target=2,
                                                     set_nodes=True)
        self.volumes = self.mesh_homogeneous.all_volumes
        self.mpfad_homogeneous = MpfaD3D(self.mesh_homogeneous)
        self.mesh_homogeneous.get_redefine_centre()
        self.mesh_homogeneous.set_global_id()


        self.mesh_heterogeneous = MeshManager(
            'meshes/geometry_two_regions_lp_test.msh', dim=3)
        self.mesh_heterogeneous.set_boundary_condition('Dirichlet',
                                                       {101: 0.0},
                                                       dim_target=2,
                                                       set_nodes=True)
        # self.mesh_heterogeneous.set_boundary_condition('Neumann',
        #                                                {201: 0.0},
        #                                                dim_target=2,
        #                                                set_nodes=True)
        self.mesh_heterogeneous.get_redefine_centre()
        self.mesh_heterogeneous.set_global_id()
        self.mpfad_heterogeneous = MpfaD3D(self.mesh_heterogeneous)
        self.hvolumes = self.mesh_heterogeneous.all_volumes

        # create Slanted Mesh

    def psol1(self, coords):
        x, y, z = coords

        return -x - 0.2 * y + z

    def lp_schneider_2018(self, coords, p_max, p_min,
                          x_max, x_min, y_max, y_min, z_max, z_min):
        x, y, z = coords
        _max = (1 / 3) * (((x_max - x)/(x_max - x_min)) +
                          ((y_max - y)/(y_max - y_min)) +
                          ((z_max - z)/(z_max - z_min))) * p_max
        _min = (1 / 3) * (((x - x_min)/(x_max - x_min)) +
                          ((y - y_min)/(y_max - y_min)) +
                          ((z - z_min)/(z_max - z_min))) * p_min
        return _max + _min




    def test_case_1(self):
        mb = self.mesh_homogeneous.mb
        """
        Test if mesh_homogeneous with tensor K_1 and solution 1
        """
        for volume in self.volumes:
            self.mesh_homogeneous.mb.tag_set_data(self.mesh_homogeneous.perm_tag,
                                                  volume, self.K_1)
        allVolumes = self.mesh_homogeneous.all_volumes
        bcVerts = self.mesh_homogeneous.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.psol1(vertCoords)
            mb.tag_set_data(self.mesh_homogeneous.dirichlet_tag, bcVert, bcVal)
        self.mpfad_homogeneous.run_solver(LPEW3(self.mesh_homogeneous).interpolate)

        for volume in allVolumes:
            coords = mb.get_coords([volume])
            u = self.psol1(coords)
            u_calc = mb.tag_get_data(self.mpfad_homogeneous.pressure_tag, volume)
            self.assertAlmostEqual(u_calc, u, delta=1e-15)

    def test_case_2(self):
        """
        Test if mesh_homogeneous with tensor K_2 and solution 1
        """
        mb = self.mesh_homogeneous.mb
        for volume in self.volumes:
            self.mesh_homogeneous.mb.tag_set_data(self.mesh_homogeneous.perm_tag,
                                                  volume, self.K_2)
        allVolumes = self.mesh_homogeneous.all_volumes
        bcVerts = self.mesh_homogeneous.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.psol1(vertCoords)
            mb.tag_set_data(self.mesh_homogeneous.dirichlet_tag, bcVert, bcVal)
        self.mpfad_homogeneous.run_solver(LPEW3(self.mesh_homogeneous).interpolate)

        for volume in allVolumes:
            coords = mb.get_coords([volume])
            u = self.psol1(coords)
            u_calc = mb.tag_get_data(self.mpfad_homogeneous.pressure_tag, volume)
            self.assertAlmostEqual(u_calc, u, delta=1e-15)

    def test_case_3(self):
        """
        Test if mesh_homogeneous with tensor K_3 and solution 1
        """
        mb = self.mesh_homogeneous.mb
        for volume in self.volumes:
            self.mesh_homogeneous.mb.tag_set_data(self.mesh_homogeneous.perm_tag,
                                                  volume, self.K_3)
        allVolumes = self.mesh_homogeneous.all_volumes
        bcVerts = self.mesh_homogeneous.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.psol1(vertCoords)
            mb.tag_set_data(self.mesh_homogeneous.dirichlet_tag, bcVert, bcVal)
        self.mpfad_homogeneous.run_solver(LPEW3(self.mesh_homogeneous).interpolate)

        for volume in allVolumes:
            coords = mb.get_coords([volume])
            u = self.psol1(coords)
            u_calc = mb.tag_get_data(self.mpfad_homogeneous.pressure_tag, volume)
            self.assertAlmostEqual(u_calc, u, delta=1e-15)

    def test_case_4(self):
        """
        Test if mesh_heterogeneous with tensor K_1/K_2 and solution 1
        """
        mb = self.mesh_heterogeneous.mb
        mtu = self.mesh_heterogeneous.mtu
        for volume in self.hvolumes:
            x, _, _ = mtu.get_average_position([volume])
            if x < 0.5:
                self.mesh_heterogeneous.mb.tag_set_data(self.mesh_heterogeneous.perm_tag,
                                                        volume, self.K_3)
            else:
                self.mesh_heterogeneous.mb.tag_set_data(self.mesh_heterogeneous.perm_tag,
                                                        volume, self.K_4)
        bcVerts = self.mesh_heterogeneous.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.psol1(vertCoords)
            mb.tag_set_data(self.mesh_heterogeneous.dirichlet_tag,
                            bcVert, bcVal)

        self.mpfad_heterogeneous.run_solver(LPEW3(self.mesh_heterogeneous).interpolate)
        error = []
        for volume in self.hvolumes:
            coords = mb.get_coords([volume])
            u = self.psol1(coords)
            u_calc = mb.tag_get_data(self.mpfad_heterogeneous.pressure_tag,
                                     volume)
            # error.append(abs((u-u_calc)/u))
        # print('max error', max(error))

            self.assertAlmostEqual(u_calc, u, delta=1e-15)

    def test_schneider_linear_preserving(self):
        """
        Test if mesh_homogeneous with tensor K_3 and solution 1
        """
        mb = self.mesh_homogeneous.mb
        for volume in self.volumes:
            self.mesh_homogeneous.mb.tag_set_data(self.mesh_homogeneous.perm_tag,
                                                  volume, self.K_3)
        allVolumes = self.mesh_homogeneous.all_volumes
        bcVerts = self.mesh_homogeneous.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.lp_schneider_2018(vertCoords, 2E5, 1E5,
                                  1., 0., 1.0, 0., 1., 0.)
            mb.tag_set_data(self.mesh_homogeneous.dirichlet_tag, bcVert, bcVal)
        self.mpfad_homogeneous.run_solver(LPEW3(self.mesh_homogeneous).interpolate)

        for volume in allVolumes:
            coords = mb.get_coords([volume])
            u = self.lp_schneider_2018(coords, 2E5, 1E5,
                                  1., 0., 1.0, 0., 1., 0.)
            u_calc = mb.tag_get_data(self.mpfad_homogeneous.pressure_tag, volume)
            self.assertAlmostEqual(u_calc[0][0], u, delta=1e-10)
