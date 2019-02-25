from benchmark_mesh_generator import BenchMeshGenerator
from mpfad.interpolation.IDW import IDW
from mpfad.interpolation.LPEW3 import LPEW3
from mpfad.interpolation.LSW import LSW
from benchmark_fvca import BenchmarkFVCA
from oblique_drain import ObliqueDrain
from discrete_maximum_principle import DiscreteMaxPrinciple

# benchmark_fvca_cases = ['1', '2', '3', '4'] #, '5', '6', '7', '8']
interpolations = [LSW] #, IDW, LSW]
# fvcaMeshesB = [BenchMeshGenerator(str(case)).generate_mesh()
#                for case in benchmark_fvca_cases]
# for mesh in fvcaMeshesB:
#     for interpolation_method in interpolations:
        # log_name_1 = ('test_case_1_' + interpolation_method.__name__ + '_'
        #               + mesh).strip('.h5m')
        # log_name_2 = ('test_case_2_' + interpolation_method.__name__ + '_'
        #               + mesh).strip('.h5m')
        # BenchmarkFVCA(mesh, interpolation_method).benchmark_case_1(log_name_1)
        # BenchmarkFVCA(mesh, interpolation_method).benchmark_case_2(log_name_2)
        # DiscreteMaxPrinciple(mesh, interpolation_method).run_lai_sheng_dmp_test()
mesh = 'meshes/benchmark_test_case_5.msh'
BenchmarkFVCA(mesh, LSW).benchmark_case_5('log_name_1')
# cases_dmp = ['8x8x8', '16x16x16', '32x32x32']  # , '16x16x16', '32x32x32', '64x64x64']
# for case in cases_dmp:
#     for im in interpolations:
#         log_name = ('monotonicity_test_' + im.__name__ + '_' + case)
#         mesh_dmp = 'meshes/monotone_' + case + '.msh'
#         DiscreteMaxPrinciple(mesh_dmp, im).run_dmp(log_name)
# test_mesh = 'meshes/mesh_darlan.msh'
# BenchmarkFVCA(test_mesh, LPEW3).benchmark_case_1('log_name_1')
# DiscreteMaxPrinciple(test_mesh, LPEW3).run_lai_sheng_dmp_test()
# meshes = {'meshes/mesh_slanted_mesh.h5m': 'coarse_mesh',
#           'meshes/oblique-drain-new.msh': 'fine_mesh'}
# for setCase, logName in meshes.items():
#     print(setCase, logName)
#     ObliqueDrain(setCase).runCase(logName)
