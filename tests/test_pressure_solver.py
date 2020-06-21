"""Mesh manager tests."""
import unittest
import numpy as np
import solvers.helpers.geometric as geo
from solvers.MpfaD import MpfaD3D
from solvers.interpolation.LPEW3 import LPEW3
from preprocessor.mesh_preprocessor import MeshManager
from pymoab import types


class MeshManagerTest(unittest.TestCase):
    """Test MeshManager class."""

    def setUp(self):
        """Init test suite."""
        K_1 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.mesh = MeshManager("meshes/mesh_test_conservative.msh", dim=3)
        self.mesh.set_media_property("Permeability", {1: K_1}, dim_target=3)
        self.mesh.set_boundary_condition(
            "Dirichlet", {102: 1.0, 101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh.get_redefine_centre()
        self.mesh.set_global_id()
        self.mpfad = MpfaD3D(self.mesh)
        self.od = MeshManager("meshes/mesh_slanted_mesh.h5m", dim=3)
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
        self.od.set_boundary_condition(
            "Dirichlet", {101: None}, dim_target=2, set_nodes=True
        )
        self.od.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=True
        )

        self.od.set_media_property(
            "Permeability",
            {1: bed_perm_isotropic, 2: fracture_perm_isotropic},
            dim_target=3,
        )
        self.od.get_redefine_centre()
        self.od.set_global_id()
        self.od_mpfad = MpfaD3D(self.od)

        self.perm = np.array([2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0])
        self.m = MeshManager("meshes/test_mesh_5_vols.h5m", dim=3)
        self.m.set_boundary_condition(
            "Dirichlet", {101: None}, dim_target=2, set_nodes=True
        )
        self.m.get_redefine_centre()
        self.m.set_global_id()
        self.m_mpfad = MpfaD3D(self.m)

    def psol1(self, coords):
        """Return solution to problem 1."""
        x, y, z = coords

        return -x - 0.2 * y

    def psol2(self, coords):
        """Return solution to problem 2."""
        x, y, z = coords

        return y ** 2

    def test_if_method_yields_exact_solution(self):
        """Test if method yields exact solution for a flow channel problem."""
        self.mtu = self.mesh.mtu
        self.mb = self.mesh.mb
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        for volume in self.mesh.all_volumes:
            c_solution = self.mb.tag_get_data(self.mpfad.pressure_tag, volume)
            a_solution = 1 - self.mb.get_coords([volume])[0]
            self.assertAlmostEqual(c_solution, a_solution, delta=1e-14)

    def test_if_inner_verts_weighted_calculation_yelds_exact_solution(self):
        """Test if inner verts weights match expected values."""
        self.mtu = self.mpfad.mtu
        self.mb = self.mpfad.mb
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        for node in self.mpfad.intern_nodes:
            analytical_solution = 1 - self.mb.get_coords([node])[0]
            nd_weights = self.mpfad.nodes_ws[node]
            p_vert = 0.0
            for volume, wt in nd_weights.items():
                p_vol = self.mpfad.mb.tag_get_data(
                    self.mpfad.pressure_tag, volume
                )
                p_vert += p_vol * wt
            self.assertAlmostEqual(p_vert, analytical_solution, delta=5e-15)

    @unittest.skip("later")
    def test_if_gradient_yields_correct_values(self):
        """Test if gradient yelds expeted values."""
        self.node_pressure_tag = self.mpfad.mb.tag_get_handle(
            "Node Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
        p_verts = []
        for node in self.mesh.all_nodes:
            p_vert = self.mpfad.mb.tag_get_data(self.mpfad.dirichlet_tag, node)
            p_verts.append(p_vert[0])
        self.mpfad.mb.tag_set_data(
            self.node_pressure_tag, self.mesh.all_nodes, p_verts
        )
        for a_volume in self.mesh.all_volumes:
            vol_faces = self.mesh.mtu.get_bridge_adjacencies(a_volume, 2, 2)
            vol_nodes = self.mesh.mtu.get_bridge_adjacencies(a_volume, 0, 0)
            vol_crds = self.mesh.mb.get_coords(vol_nodes)
            vol_crds = np.reshape(vol_crds, ([4, 3]))
            vol_volume = self.mesh.get_tetra_volume(vol_crds)
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
            v = 0.0
            for face in vol_faces:
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
                v += vel * np.sqrt(np.dot(area_vect, area_vect))

    def test_if_flux_is_conservative_for_all_volumes(self):
        """Test if flux is conservative for all volumes in the test domain."""
        mb = self.od.mb
        bcVerts = self.od.get_boundary_nodes()
        for bcVert in bcVerts:
            vertCoords = mb.get_coords([bcVert])
            bcVal = self.psol1(vertCoords)
            mb.tag_set_data(self.od.dirichlet_tag, bcVert, bcVal)

        self.node_pressure_tag = self.od_mpfad.mb.tag_get_handle(
            "Node Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )
        self.od_mpfad.run_solver(LPEW3(self.od).interpolate)
        p_verts = []
        for node in self.od.all_nodes:
            p_vert = self.od_mpfad.mb.tag_get_data(
                self.od_mpfad.dirichlet_tag, node
            )
            p_verts.append(p_vert[0])
        self.od_mpfad.mb.tag_set_data(
            self.node_pressure_tag, self.od.all_nodes, p_verts
        )
        for a_volume in self.od.all_volumes:
            vol_centroid = self.od.mtu.get_average_position([a_volume])
            vol_faces = self.od.mtu.get_bridge_adjacencies(a_volume, 2, 2)
            vol_p = self.od.mb.tag_get_data(
                self.od_mpfad.pressure_tag, a_volume
            )[0][0]
            vol_nodes = self.od.mtu.get_bridge_adjacencies(a_volume, 0, 0)
            vol_crds = self.od.mb.get_coords(vol_nodes)
            vol_crds = np.reshape(vol_crds, ([4, 3]))
            vol_volume = self.od.get_tetra_volume(vol_crds)
            vol_perm = self.od.mb.tag_get_data(
                self.od_mpfad.perm_tag, a_volume
            ).reshape([3, 3])
            fluxes = []
            for a_face in vol_faces:
                f_nodes = self.od.mtu.get_bridge_adjacencies(a_face, 0, 0)
                fc_nodes = self.od.mb.get_coords(f_nodes)
                fc_nodes = np.reshape(fc_nodes, ([3, 3]))
                grad = np.zeros(3)
                for i in range(len(fc_nodes)):
                    set_of_verts = np.array(
                        [fc_nodes[i], fc_nodes[i - 1], vol_centroid]
                    )
                    area_vect = geo._area_vector(
                        set_of_verts, fc_nodes[i - 2]
                    )[0]
                    p_node_op = self.od_mpfad.mb.tag_get_data(
                        self.node_pressure_tag, f_nodes[i - 2]
                    )[0][0]
                    grad += area_vect * p_node_op
                area_vect = geo._area_vector(fc_nodes, vol_centroid)[0]
                grad += area_vect * vol_p
                grad = grad / (3.0 * vol_volume)
                flux = -np.dot(np.dot(vol_perm, grad), area_vect)
                fluxes.append(flux)
            fluxes_sum = abs(sum(fluxes))
            self.assertAlmostEqual(fluxes_sum, 0.0, delta=1e-9)

    @unittest.skip("later")
    def test_if_method_yields_correct_T_matrix(self):
        """Test not well suited."""
        for node in self.m.get_boundary_nodes():
            coords = self.m.mb.get_coords([node])
            g_D = coords[1] ** 2
            self.m.mb.tag_set_data(self.m.dirichlet_tag, node, g_D)
        volumes = self.m.all_volumes
        source = [
            -0.666666721827,
            -0.666666721827,
            -0.666667091901,
            -0.666667091901,
            -1.33333307358,
        ]
        c = 0
        for volume in volumes:
            self.m.mb.tag_set_data(self.m.perm_tag, volume, self.perm)
            self.m.mb.tag_set_data(self.m.source_tag, volume, source[c])
            c += 1
        self.m_mpfad.run_solver(LPEW3(self.m).interpolate)

    def test_mobility_tag_is_created_in_single_phase(self):
        self.m_mpfad.get_mobility()
        for face in self.m_mpfad.intern_faces:
            self.assertEqual(
                self.m_mpfad.mb.tag_get_data(
                    self.m_mpfad.face_mobility_tag, face
                )[0][0],
                1.0,
            )
