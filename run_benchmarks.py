from benchmark_mesh_generator import BenchMeshGenerator
from benchmark_fvca import BenchmarkFVCA
from oblique_drain import ObliqueDrain

# benchmark_fvca_cases = ['1', '2', '3', '4', '5', '6', '7', '8']
benchmark_fvca_cases = [None]

try:

    fvcaMeshesB = [BenchMeshGenerator(str(case)).generate_mesh()
                   for case in benchmark_fvca_cases]

    for mesh in fvcaMeshesB:
        log_name_1 = ('Report_benchmark_FVCA_test_case_1_'
                      + mesh).strip('.h5m')
        # log_name_2 = ('Report_benchmark_FVCA_test_case_2_'
        #               + mesh).strip('.h5m')
        BenchmarkFVCA(mesh).benchmark_case_1(log_name_1)
        # BenchmarkFVCA(mesh).benchmark_case_2(log_name_2)
except:
    pass

# BenchmarkFVCA('test_mesh_5_vols.h5m').benchmark_case_1('debug_tests')


path = 'paper3D-resultados/linear_preserving/pp-data-files'

# obliqueDrainMeshes = [path + '/oblique-drain.msh' for i in range(1, 2)]
mesh = 'meshes/oblique-drain.msh'
ObliqueDrain(mesh, 0.2).runCase('oblique_drain_log')
# ObliqueDrain('oblique-drain.msh', 0.2).runCase('log_name_debugging')
# for idx, mesh in zip(range(1, 4), obliqueDrainMeshes):
#     log_name = 'Results_Oblique_drain_' + str(idx)
#     ObliqueDrain(mesh, 0.2).runCase(log_name)
