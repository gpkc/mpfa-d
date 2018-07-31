import unittest
from mesh_generator import GenerateMesh


class MeshGeneratorTest(unittest.TestCase):

    def setUp(self):
        path = 'mesh_bench/meshB_tetra/tet.2.msh'
        self.mesh_tetra00 = GenerateMesh(path)

    def tearDown(self):
        self.mesh_tetra00 = None

    """
    def test_mesh_reading_tool(self):
        information_set = self.mesh_tetra00.locate_information()
        self.assertEqual(information_set, [10, 12, 18, 45, 90, 135, 248])

    def test_method_reads_values_properly(self):
        information_set = [10, 12, 18, 45, 90, 135, 248]
        f = open('tet.00.msh')
        lines = f.readlines()
        self.assertEqual(lines[information_set[0] - 1].strip('          \n'),
                         '26')
        f.close()

    def test_method_gets_all_vertices(self):
        vertices = self.mesh_tetra00.get_all_vertices()
        self.assertEqual(len(vertices), 26)

    def test_if_a_vert_is_returned(self):
        vertices = self.mesh_tetra00.get_all_vertices()
        a_vertice_coords = self.mesh_tetra00.mb.get_coords([vertices[0]])
        for i in range(0, 3):
            self.assertEqual(a_vertice_coords[i], 0.0)

    def test_if_volumes_are_created(self):
        self.mesh_tetra00.create_tags()
        self.mesh_tetra00.get_all_vertices()
        self.mesh_tetra00.create_volumes()
        all_volumes = self.mesh_tetra00.mb.get_entities_by_dimension(0, 3)
        self.assertEqual(len(all_volumes), 44)

    def test_if_faces_are_created(self):
        self.mesh_tetra00.create_tags()
        self.mesh_tetra00.get_all_vertices()
        self.mesh_tetra00.create_volumes()
        volumes = self.mesh_tetra00.mb.get_entities_by_dimension(0, 3)
        faces = self.mesh_tetra00.mtu.get_bridge_adjacencies(volumes, 3, 2)
        self.assertEqual(len(faces), 112)

    def test_boundaries_are_created(self):
        self.mesh_tetra00.create_tags()
        self.mesh_tetra00.get_all_vertices()
        self.mesh_tetra00.create_volumes()
        self.mesh_tetra00.create_dirichlet_boundary_conditions()

    """
    def test_if_mesh_works_in_mesh_preprocessor(self):
        self.mesh_tetra00.create_tags()
        self.mesh_tetra00.get_all_vertices()
        self.mesh_tetra00.create_volumes()
        self.mesh_tetra00.create_dirichlet_boundary_conditions()
        self.mesh_tetra00.write_msh_file('mesh_tet2.h5m')
