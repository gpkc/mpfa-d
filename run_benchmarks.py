from benchmark_mesh_generator import BenchMeshGenerator
from benchmark_fvca import BenchmarkFVCA
# from oblique_drain import ObliqueDrain

benchmark_fvca_cases = ['1'] #, '2', '3', '4', '5', '6', '7', '8']
# benchmark_fvca_cases = [None]

try:

    fvcaMeshesB = [BenchMeshGenerator(str(case)).generate_mesh()
                   for case in benchmark_fvca_cases]

    for mesh in fvcaMeshesB:
        log_name_1 = ('Report_benchmark_FVCA_test_case_1_'
                      + mesh).strip('.h5m')
        log_name_2 = ('Report_benchmark_FVCA_test_case_2_'
                      + mesh).strip('.h5m')
        BenchmarkFVCA(mesh).benchmark_case_1(log_name_1)
        BenchmarkFVCA(mesh).benchmark_case_2(log_name_2)
except:
    pass
# meshes = {'meshes/mesh_slanted_mesh.h5m': 'coarse_mesh',
#           'meshes/oblique-drain-new.msh': 'fine_mesh'}
# for setCase, logName in meshes.items():
#     print(setCase, logName)
#     ObliqueDrain(setCase).runCase(logName)
