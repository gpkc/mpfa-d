from benchmark_mesh_generator import BenchMeshGenerator
from benchmark_fvca import BenchmarkFVCA


# mesh_000 = BenchMeshGenerator('000').generate_mesh()
# mesh_00 = BenchMeshGenerator('00').generate_mesh()
mesh_0 = BenchMeshGenerator('0').generate_mesh()
# mesh_1 = BenchMeshGenerator('1').generate_mesh()
# mesh_2 = BenchMeshGenerator('2').generate_mesh()
# mesh_3 = BenchMeshGenerator('3').generate_mesh()
# mesh_4 = BenchMeshGenerator('4').generate_mesh()
# mesh_5 = BenchMeshGenerator('5').generate_mesh()
# mesh_6 = BenchMeshGenerator('6').generate_mesh()

meshes = [mesh_0]

for mesh in meshes:
    log_name_1 = 'Bench_1_' + mesh
    # log_name_2 = 'Bench_2_' + mesh

    BenchmarkFVCA(mesh).benchmark_case_1(log_name_1)
    # BenchmarkFVCA(mesh).benchmark_case_2(log_name_2)
