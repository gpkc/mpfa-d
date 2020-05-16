import os

from preprocessor.benchmark_mesh_generator import BenchMeshGenerator

from mpfad.interpolation.IDW import IDW
from solvers.interpolation.LPEW3 import LPEW3
from mpfad.interpolation.LSW import LSW

from single_phase_cases.benchmark_fvca import BenchmarkFVCA

from single_phase_cases.oblique_drain import ObliqueDrain
from single_phase_cases.discrete_maximum_principle import DiscreteMaxPrinciple
from single_phase_cases.mpfad_mge_tests import TestCasesMGE

interpolation_methods = [LPEW3, IDW, LSW]
benchmark_fvca_cases = [1, 2, 3, 4, 5, 7, 8]
fvcaMeshesB = [
    BenchMeshGenerator(str(case)).generate_mesh()
    for case in benchmark_fvca_cases
]
for mesh in fvcaMeshesB:
    for im in interpolation_methods:
        im_name = im.__name__
        log_name_1 = ("_case_1_" + im_name + "_" + mesh).replace(".h5m", "")
        log_name_2 = ("_case_2_" + im_name + "_" + mesh).replace(".h5m", "")
        log_name_3 = ("_case_3_" + im_name + "_" + mesh).replace(".h5m", "")
        log_name_5 = ("_case_5_" + im_name + "_" + mesh).replace(".h5m", "")
        BenchmarkFVCA(mesh, im).benchmark_case_1(log_name_1)
        BenchmarkFVCA(mesh, im).benchmark_case_2(log_name_2)
        BenchmarkFVCA(mesh, im).benchmark_case_3(log_name_3)
        BenchmarkFVCA(mesh, im).benchmark_case_5(log_name_5)
        TestCasesMGE(mesh, im).run_case(log_name_1, "mge_test_case_1")
        TestCasesMGE(mesh, im).run_case(log_name_1, "mge_test_case_2")
        TestCasesMGE(mesh, im).run_case(log_name_1, "mge_test_case_3")
        TestCasesMGE(mesh, im).run_case(log_name_1, "mge_test_case_4")
        os.remove(mesh)

meshes = []
cases_dmp = ["8x8x8", "16x16x16", "32x32x32", "64x64x64"]
for case in cases_dmp:
    for im in interpolation_methods:
        log_name = "monotonicity_test_" + im.__name__ + "_" + case
        mesh_dmp = "meshes/monotone_" + case + ".msh"
        DiscreteMaxPrinciple(mesh_dmp, im).run_dmp(log_name)
        DiscreteMaxPrinciple(mesh_dmp, im).run_lai_sheng_dmp_test()

meshes = {
    "meshes/oblique-drain-new.msh": "distort",
    "meshes/mesh_slanted_mesh.h5m": "coarse_mesh",
}
for setCase, logName in meshes.items():
    for im in interpolation_methods:
        print(setCase, logName)
        ObliqueDrain(setCase).runCase(im, logName)
