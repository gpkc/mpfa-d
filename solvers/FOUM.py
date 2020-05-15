# from pymoab import types
# import mpfad.helpers.geometric as geo
# import numpy as np
# from mpfad.MpfaD import MpfaD3D
# from mpfad.interpolation.LPEW3 import LPEW3
from preprocessor.mesh_preprocessor import MeshManager


class TwoPhaseFlow:
    def __init__(
        self, filename, fluid_props, numericals
    ):  # fluid_props = iOilSat, iWaterSat, visOil, visWater
        self.mesh = MeshManager(filename, dim=3)
        self.mesh.set_boundary_condition(
            "WaterSat", {1: 0.2}, dim_target=3, set_nodes=True
        )

    def add_wells(
        self, coords, operation, operatePressure=None, operateFlux=None
    ):
        pass

    def get_fluid_relative_perm(self, sw_field, volumes):
        pass

    def calc_brooks_korey_rel_perm(self, sw, a, b):
        pass

    def calc_mobility(self, kr):
        pass
