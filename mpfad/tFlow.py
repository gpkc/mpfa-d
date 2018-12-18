from pymoab import types
import mpfad.helpers.geometric as geo
import numpy as np
from mpfad.MpfaD import MpfaD3D
from mpfad.interpolation.LPEW3 import LPEW3
from mesh_preprocessor import MeshManager

class tFlow:

    def __init__(self, mesh_data, fluid_props, numericals):  # fluid_props = iOilSat, iWaterSat, visOil, visWater
        self.mesh = MeshManager(filename, dim=3)
        self.mesh.set_boundary_condition('Dirichlet', {101: 0.0},
                                         dim_target=2, set_nodes=True)
        self.mesh.set_global_id()
        self.mesh.get_redefine_centre()
        self.mpfad = MpfaD3D(self.mesh)
        self.lpew3 = LPEW3(self.mesh)

    def add_wells(self, coords, operation, operatePressure=0, operateFlux=0):
        pass

    def get_fluid_relative_perm(self, sw_field, volumes):
        pass

    def calc_brooks_korey_rel_perm(self, sw, a, b):
        pass
