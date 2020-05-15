from preprocessor.mesh_generator import GenerateMesh


class BenchMeshGenerator:
    def __init__(self, mesh):
        self.mesh = mesh
        path = "mesh_bench/meshB_tetra/tet_" + self.mesh + ".msh"
        self.mesh_tetra = GenerateMesh(path)

    def generate_mesh(self):
        self.mesh_tetra.create_tags()
        self.mesh_tetra.get_all_vertices()
        self.mesh_tetra.create_volumes()
        self.mesh_tetra.create_dirichlet_boundary_conditions()
        mesh_file = "mesh_tet" + self.mesh + ".h5m"
        self.mesh_tetra.write_msh_file(mesh_file)
        self.mesh_tetra = None
        return mesh_file
