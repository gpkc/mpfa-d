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
        self.K_2 = np.array([100.0, 1.0, 0.0,
                             1.0, 100.0, 1.0,
                             0.0, 1.0, 100.0])
        self.k_3 = np.array([1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1E3])
        self.k_4 = np.array([1.0, 0.0, 0.0,
                             0.0, 1E3, 0.0,
                             0.0, 0.0, 1.0])

        self.mesh_homogeneous = MeshManager('test_mesh_5_vols.h5m', dim=3)
        self.mesh_homogeneous.set_boundary_condition('Dirichlet',
                                                     {101: 0.0},
                                                     dim_target=2,
                                                     set_nodes=True)
        # self.mesh_homogeneous.set_boundary_condition('Neumann', {201: 0.0},
        #                                              dim_target=2,
        #                                              set_nodes=True)
        self.volumes = self.mesh_homogeneous.all_volumes
        self.mpfad_homogeneous = MpfaD3D(self.mesh_homogeneous)
        self.mesh_homogeneous.get_redefine_centre()
        self.mesh_homogeneous.set_global_id()


        self.mesh_heterogeneous = MeshManager(
            'meshes/lp_heterogeneous.msh', dim=3)
        self.mesh_heterogeneous.set_boundary_condition('Dirichlet',
                                                       {101: 0.0},
                                                       dim_target=2,
                                                       set_nodes=True)
        self.mesh_heterogeneous.set_boundary_condition('Neumann',
                                                       {201: 0.0},
                                                       dim_target=2,
                                                       set_nodes=True)
        self.mesh_heterogeneous.get_redefine_centre()
        self.mesh_heterogeneous.set_global_id()
        self.mpfad_heterogeneous = MpfaD3D(self.mesh_heterogeneous)

    def psol1(self, coords):
        x, y, z = coords
        return -x - 0.2 * y

    def psol2(self, coords):
        x, y, z = coords
        return 2 * x + 3 * y - z

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
            self.assertAlmostEqual(
                u_calc, u, delta=1e-15)

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
            self.assertAlmostEqual(
                u_calc, u, delta=1e-15)

    def test_case_3(self):
        """
        Test if mesh_homogeneous with tensor K_5 and solution 1
        """
        pass

    def test_case_4(self):
        """
        Test if mesh_heterogeneous with tensor K_1/K_2 and solution 1
        """
        pass

    def test_case_5(self):
        """
        Test if mesh_heterogeneous with tensor K_3/K_4 and solution 1
        """
        pass

    def test_case_6(self):
        """
        Test if mesh_heterogeneous with tensor K_5/K_6 and solution 1
        """
        pass

    def test_case_7(self):
        """
        Test if mesh_homogeneous with tensor K_1 and solution 2
        """
        pass

    def test_case_8(self):
        """
        Test if mesh_homogeneous with tensor K_3 and solution 2
        """
        pass

    def test_case_9(self):
        """
        Test if mesh_homogeneous with tensor K_5 and solution 2
        """
        pass

    def test_case_10(self):
        """
        Test if mesh_heterogeneous with tensor K_1/K_2 and solution 2
        """
        pass

    def test_case_11(self):
        """
        Test if mesh_heterogeneous with tensor K_3/K_4 and solution 2
        """
        pass

    def test_case_12(self):
        """
        Test if mesh_heterogeneous with tensor K_5/K_6 and solution 2
        """
        pass
