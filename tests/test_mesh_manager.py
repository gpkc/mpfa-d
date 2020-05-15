import unittest
import numpy as np
from mpfad.MpfaD import MpfaD3D
from mpfad.interpolation.LPEW3 import LPEW3
from mesh_preprocessor import MeshManager
from pymoab import types
import mpfad.helpers.geometric as geo


class MeshManagerTest(unittest.TestCase):

    """
    These tests are more MeshManager class related. We should rename
    this test suite.
    """

    def setUp(self):

        K_1 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        K_2 = np.array([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0])

        self.mesh_1 = MeshManager("meshes/mesh_test_1.msh")

        self.mesh_2 = MeshManager("meshes/mesh_test_2.msh", dim=3)
        self.mesh_2.set_media_property("Permeability", {1: K_1}, dim_target=3)
        self.mesh_2.set_boundary_condition(
            "Dirichlet", {102: 1.0, 101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_2.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_2.set_global_id()
        self.mesh_2.get_redefine_centre()
        self.mpfad_2 = MpfaD3D(self.mesh_2)

        self.mesh_3 = MeshManager(
            "meshes/geometry_two_regions_test.msh", dim=3
        )
        self.mesh_3.set_media_property(
            "Permeability", {1: K_1, 2: K_2}, dim_target=3
        )
        self.mesh_3.set_boundary_condition(
            "Dirichlet", {102: 1.0, 101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_3.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh_3.get_redefine_centre()
        self.mpfad_3 = MpfaD3D(self.mesh_3)

        self.mesh_4 = MeshManager("meshes/mesh_darlan.msh")
        self.mesh_4.set_global_id()
        self.mesh_4.get_redefine_centre()

    # @unittest.skip('no need for testing')
    def test_if_method_has_all_dirichlet_nodes(self):
        self.assertEqual(len(self.mpfad_3.dirichlet_nodes), 10)

    # @unittest.skip('no need for testing')
    def test_if_method_has_all_neumann_nodes(self):
        self.assertEqual(len(self.mpfad_3.neumann_nodes), 12)

    # @unittest.skip('no need for testing')
    def test_if_method_has_all_intern_nodes(self):
        self.assertEqual(len(self.mpfad_3.intern_nodes), 1)

    # @unittest.skip('no need for testing')
    def test_if_method_has_all_dirichlet_faces(self):
        self.assertEqual(len(self.mpfad_2.dirichlet_faces), 4)

    # @unittest.skip('no need for testing')
    def test_if_method_has_all_neumann_faces(self):
        self.assertEqual(len(self.mpfad_2.neumann_faces), 8)

    # @unittest.skip('no need for testing')
    def test_if_method_has_all_intern_faces(self):
        self.assertEqual(len(self.mpfad_2.intern_faces), 18)

    # @unittest.skip('no need for testing')
    def test_get_tetra_volume(self):
        a_tetra = self.mesh_1.all_volumes[0]
        tetra_nodes = self.mesh_1.mb.get_adjacencies(a_tetra, 0)
        tetra_nodes_coords = self.mesh_1.mb.get_coords(tetra_nodes)
        tetra_nodes_coords = np.reshape(tetra_nodes_coords, (4, 3))
        vol_eval = self.mesh_1.get_tetra_volume(tetra_nodes_coords)
        self.assertAlmostEqual(vol_eval, 1 / 12.0, delta=1e-15)

    @unittest.skip("skip this test")
    def test_get_proper_vert(self):
        verts = []
        vectors = []
        geometric = []
        id_volumes = []
        left_volumes = []
        for face in self.mesh_2.dirichlet_faces:
            I, J, K = self.mesh_2.mtu.get_bridge_adjacencies(face, 2, 0)

            left_volume = np.asarray(
                self.mesh_2.mtu.get_bridge_adjacencies(face, 2, 3),
                dtype="uint64",
            )
            id_volume = self.mesh_2.mb.tag_get_data(
                self.mpfad_2.global_id_tag, left_volume
            )[0][0]
            left_volumes.append(left_volume)
            id_volumes.append(id_volume)

            JI = self.mesh_2.mb.get_coords([I]) - self.mesh_2.mb.get_coords(
                [J]
            )
            JK = self.mesh_2.mb.get_coords([K]) - self.mesh_2.mb.get_coords(
                [J]
            )
            LJ = (
                self.mesh_2.mb.get_coords([J])
                - self.mesh_2.mb.tag_get_data(
                    self.mesh_2.volume_centre_tag, left_volume
                )[0]
            )
            N_IJK = np.cross(JI, JK) / 2.0
            test = np.dot(LJ, N_IJK)
            if test < 0.0:
                I, K = K, I
                JI = self.mesh_2.mb.get_coords(
                    [I]
                ) - self.mesh_2.mb.get_coords([J])
                JK = self.mesh_2.mb.get_coords(
                    [K]
                ) - self.mesh_2.mb.get_coords([J])
                N_IJK = np.cross(JI, JK) / 2.0
            tan_JI = np.cross(N_IJK, JI)
            print("qwerqwer", tan_JI)
            tan_JK = np.cross(N_IJK, JK)
            face_area = np.sqrt(np.dot(N_IJK, N_IJK))
            h_L = geo.get_height(N_IJK, LJ)

            verts.append([I, J, K])
            vectors.append([JI, JK, N_IJK, tan_JI, tan_JK])
            geometric.append([h_L, face_area])
        I = np.asarray(
            [verts[j][0] for j in range(len(self.mesh_2.dirichlet_faces))],
            dtype="uint64",
        )
        J = np.asarray(
            [verts[j][1] for j in range(len(self.mesh_2.dirichlet_faces))],
            dtype="uint64",
        )
        K = np.asarray(
            [verts[j][2] for j in range(len(self.mesh_2.dirichlet_faces))],
            dtype="uint64",
        )

        JI = np.asarray(
            [vectors[j][0] for j in range(len(self.mesh_2.dirichlet_faces))],
            dtype="uint64",
        ).reshape([len(self.mesh_2.dirichlet_faces), 3])
        print(JI)
        JK = np.asarray(
            [vectors[j][1] for j in range(len(self.mesh_2.dirichlet_faces))],
            dtype="uint64",
        ).reshape([len(self.mesh_2.dirichlet_faces), 3])
        N_IJK = np.asarray(
            [vectors[j][2] for j in range(len(self.mesh_2.dirichlet_faces))],
            dtype="uint64",
        ).reshape([len(self.mesh_2.dirichlet_faces), 3])
        tan_JI = np.asarray(
            [vectors[j][3] for j in range(len(self.mesh_2.dirichlet_faces))],
            dtype="uint64",
        ).reshape([len(self.mesh_2.dirichlet_faces), 3])
        tan_JK = np.asarray(
            [vectors[j][4] for j in range(len(self.mesh_2.dirichlet_faces))],
            dtype="uint64",
        ).reshape([len(self.mesh_2.dirichlet_faces), 3])
        h_L = np.asarray(
            [geometric[j][0] for j in range(len(self.mesh_2.dirichlet_faces))],
            dtype="uint64",
        )
        face_area = np.asarray(
            [geometric[j][1] for j in range(len(self.mesh_2.dirichlet_faces))],
            dtype="uint64",
        )
        g_I = self.mpfad_2.mb.tag_get_data(self.mesh_2.dirichlet_tag, I)
        g_J = self.mpfad_2.mb.tag_get_data(self.mesh_2.dirichlet_tag, J)
        g_K = self.mpfad_2.mb.tag_get_data(self.mesh_2.dirichlet_tag, K)

        dot_term = np.dot(-tan_JK, LJ) / (2 * h_L * face_area)
        print(tan_JK, face_area, dot_term)
        # cdf_term = h1 * S * Kt1
        # D_JK = (dot_term + cdf_term) / (2 * h1 * S)
