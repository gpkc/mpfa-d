"""Tests for linearity preserving solutions."""
import unittest
import numpy as np
from solvers.nMpfaD import MpfaD3D
from solvers.interpolation.LPEW3 import LPEW3
from preprocessor.mesh_preprocessor import MeshManager


class LinearityPreservingTests(unittest.TestCase):
    """Linear preserving test suite."""

    def setUp(self):
        """Init test suite."""
        self.K_1 = np.array([1.0, 0.50, 0.0, 0.50, 1.0, 0.50, 0.0, 0.50, 1.0])
        self.K_2 = np.array([10.0, 1.0, 0.0, 1.0, 10.0, 1.0, 0.0, 1.0, 10.0])
        self.K_3 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-3])
        self.K_4 = np.array([1.0, 0.0, 0.0, 0.0, 1e3, 0.0, 0.0, 0.0, 1.0])

        bed_perm_isotropic = [
            0.965384615384615,
            0.173076923076923,
            0.0,
            0.173076923076923,
            0.134615384615385,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        fracture_perm_isotropic = [
            96.538461538461530,
            17.307692307692307,
            0.0,
            17.307692307692307,
            13.461538461538462,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

        self.mesh_homogeneous = MeshManager(
            "meshes/test_mesh_5_vols.h5m", dim=3
        )
        self.mesh_homogeneous.set_boundary_condition(
            "Dirichlet", {101: 0.0}, dim_target=2, set_nodes=True
        )
        self.volumes = self.mesh_homogeneous.all_volumes
        self.mpfad_homogeneous = MpfaD3D(self.mesh_homogeneous)
        self.mesh_homogeneous.get_redefine_centre()
        self.mesh_homogeneous.set_global_id()

        self.mesh_heterogeneous = MeshManager(
            "meshes/geometry_two_regions_lp_test.msh", dim=3
        )
        self.mesh_heterogeneous.set_boundary_condition(
            "Dirichlet", {101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_heterogeneous.get_redefine_centre()
        self.mesh_heterogeneous.set_global_id()
        self.mpfad_heterogeneous = MpfaD3D(self.mesh_heterogeneous)
        self.hvolumes = self.mesh_heterogeneous.all_volumes

        self.slanted_mesh = MeshManager("meshes/mesh_slanted_mesh.h5m", dim=3)
        self.slanted_mesh.set_boundary_condition(
            "Dirichlet", {101: None}, dim_target=2, set_nodes=True
        )
        self.slanted_mesh.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )

        self.slanted_mesh.set_media_property(
            "Permeability",
            {1: bed_perm_isotropic, 2: fracture_perm_isotropic},
            dim_target=3,
        )
        self.slanted_mesh.get_redefine_centre()
        self.slanted_mesh.set_global_id()
        self.mpfad_slanted_mesh = MpfaD3D(self.slanted_mesh)

    def psol1(self, coords):
        """Return solution for test case 1."""
        x, y, z = coords

        return -x - 0.2 * y

    def lp_schneider_2018(
        self, coords, p_max, p_min, x_max, x_min, y_max, y_min, z_max, z_min
    ):
        """Return the solution for the Schineider (2018) example."""
        x, y, z = coords
        _max = (
            (1 / 3)
            * (
                ((x_max - x) / (x_max - x_min))
                + ((y_max - y) / (y_max - y_min))
                + ((z_max - z) / (z_max - z_min))
            )
            * p_max
        )
        _min = (
            (1 / 3)
            * (
                ((x - x_min) / (x_max - x_min))
                + ((y - y_min) / (y_max - y_min))
                + ((z - z_min) / (z_max - z_min))
            )
            * p_min
        )
        return _max + _min

    def test_case_1(self):
        """Run the test case 1."""
        mb = self.mesh_homogeneous.mb
        for volume in self.volumes:
            mb.tag_set_data(self.mesh_homogeneous.perm_tag, volume, self.K_1)
        allVolumes = self.mesh_homogeneous.all_volumes
        bcVerts = self.mesh_homogeneous.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.psol1(vertCoords)
            mb.tag_set_data(self.mesh_homogeneous.dirichlet_tag, bcVert, bcVal)
        self.mpfad_homogeneous.run_solver(
            LPEW3(self.mesh_homogeneous).interpolate
        )
        import pdb; pdb.set_trace()
        for volume in allVolumes:
            coords = mb.get_coords([volume])
            u = self.psol1(coords)
            u_calc = mb.tag_get_data(
                self.mpfad_homogeneous.pressure_tag, volume
            )
            self.assertAlmostEqual(u_calc, u, delta=1e-15)

    def test_case_2(self):
        """Test if mesh_homogeneous with tensor K_2 and solution 1."""
        mb = self.mesh_homogeneous.mb
        for volume in self.volumes:
            mb.tag_set_data(self.mesh_homogeneous.perm_tag, volume, self.K_2)
        allVolumes = self.mesh_homogeneous.all_volumes
        bcVerts = self.mesh_homogeneous.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.psol1(vertCoords)
            mb.tag_set_data(self.mesh_homogeneous.dirichlet_tag, bcVert, bcVal)
        self.mpfad_homogeneous.run_solver(
            LPEW3(self.mesh_homogeneous).interpolate
        )

        for volume in allVolumes:
            coords = mb.get_coords([volume])
            u = self.psol1(coords)
            u_calc = mb.tag_get_data(
                self.mpfad_homogeneous.pressure_tag, volume
            )
            self.assertAlmostEqual(u_calc, u, delta=1e-15)

    def test_case_3(self):
        """Test if mesh_homogeneous with tensor K_3 and solution 1."""
        mb = self.mesh_homogeneous.mb
        for volume in self.volumes:
            mb.tag_set_data(self.mesh_homogeneous.perm_tag, volume, self.K_3)
        allVolumes = self.mesh_homogeneous.all_volumes
        bcVerts = self.mesh_homogeneous.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.psol1(vertCoords)
            mb.tag_set_data(self.mesh_homogeneous.dirichlet_tag, bcVert, bcVal)
        self.mpfad_homogeneous.run_solver(
            LPEW3(self.mesh_homogeneous).interpolate
        )

        for volume in allVolumes:
            coords = mb.get_coords([volume])
            u = self.psol1(coords)
            u_calc = mb.tag_get_data(
                self.mpfad_homogeneous.pressure_tag, volume
            )
            self.assertAlmostEqual(u_calc, u, delta=1e-15)

    def test_case_4(self):
        """Test if mesh_heterogeneous with tensor K_1/K_2 and solution 1."""
        mb = self.mesh_heterogeneous.mb
        mtu = self.mesh_heterogeneous.mtu
        for volume in self.hvolumes:
            x, _, _ = mtu.get_average_position([volume])
            if x < 0.5:
                mb.tag_set_data(
                    self.mesh_heterogeneous.perm_tag, volume, self.K_3
                )
            else:
                mb.tag_set_data(
                    self.mesh_heterogeneous.perm_tag, volume, self.K_4
                )
        bcVerts = self.mesh_heterogeneous.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.psol1(vertCoords)
            mb.tag_set_data(
                self.mesh_heterogeneous.dirichlet_tag, bcVert, bcVal
            )

        self.mpfad_heterogeneous.run_solver(
            LPEW3(self.mesh_heterogeneous).interpolate
        )

        for volume in self.hvolumes:
            coords = mb.get_coords([volume])
            u = self.psol1(coords)
            u_calc = mb.tag_get_data(
                self.mpfad_heterogeneous.pressure_tag, volume
            )

            self.assertAlmostEqual(u_calc, u, delta=1e-15)

    def test_schneider_linear_preserving(self):
        """Test if mesh_homogeneous with tensor K_3 and solution 1."""
        mb = self.mesh_homogeneous.mb
        for volume in self.volumes:
            mb.tag_set_data(self.mesh_homogeneous.perm_tag, volume, self.K_3)
        allVolumes = self.mesh_homogeneous.all_volumes
        bcVerts = self.mesh_homogeneous.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.lp_schneider_2018(
                vertCoords, 2e5, 1e5, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0
            )
            mb.tag_set_data(self.mesh_homogeneous.dirichlet_tag, bcVert, bcVal)
        self.mpfad_homogeneous.run_solver(
            LPEW3(self.mesh_homogeneous).interpolate
        )

        for volume in allVolumes:
            coords = mb.get_coords([volume])
            u = self.lp_schneider_2018(
                coords, 2e5, 1e5, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0
            )
            u_calc = mb.tag_get_data(
                self.mpfad_homogeneous.pressure_tag, volume
            )
            self.assertAlmostEqual(u_calc[0][0], u, delta=1e-10)

    def test_oblique_drain_contains_all_faces(self):
        """Test case: the Oblique Drain (linear solution)."""
        all_faces = self.slanted_mesh.all_faces
        self.assertEqual(len(all_faces), 44)

    def test_if_mesh_contains_all_dirichlet_faces(self):
        """Test if mesh contains all Dirichlet faces."""
        dirichlet_faces = self.slanted_mesh.dirichlet_faces
        self.assertEqual(len(dirichlet_faces), 16)

    def test_if_mesh_contains_all_neumann_faces(self):
        """Test if mesh contains all neumann faces."""
        neumann_faces = self.slanted_mesh.neumann_faces
        self.assertEqual(len(neumann_faces), 12)

    def test_if_neumann_bc_is_appplied(self):
        """Test if newmann BC is applied."""
        for face in self.slanted_mesh.neumann_faces:
            face_flow = self.slanted_mesh.mb.tag_get_data(
                self.slanted_mesh.neumann_tag, face
            )[0][0]
            self.assertEqual(face_flow, 0.0)

    def test_oblique_drain(self):
        """Test with slanted_mesh."""
        mb = self.slanted_mesh.mb
        allVolumes = self.slanted_mesh.all_volumes
        bcVerts = self.slanted_mesh.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.psol1(vertCoords)
            mb.tag_set_data(self.slanted_mesh.dirichlet_tag, bcVert, bcVal)

        self.mpfad_slanted_mesh.run_solver(
            LPEW3(self.slanted_mesh).interpolate
        )

        for volume in allVolumes:
            coords = mb.get_coords([volume])
            u = self.psol1(coords)
            u_calc = mb.tag_get_data(
                self.mpfad_slanted_mesh.pressure_tag, volume
            )
            self.assertAlmostEqual(u_calc[0][0], u, delta=1e-13)
