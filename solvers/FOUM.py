# import numpy as np


class TwoPhaseFlow:
    def __init__(self, mesh_data, fluid_props):
        self.mesh_data = mesh_data
        self.mb = mesh_data.mb
        self.mtu = mesh_data.mtu
        self.fluid_props = fluid_props
        # INIT DEPENDECIES AND VARS
        # INIT SAT
        # INIT VELOCITY FIELD
        # INIT SPECIFIC MASS
        # INIT VISCOSITY

    def fractional_flux(self, s_W):
        f = s_W / (1 - s_W)
        return f

    def kr(self, s_W, krw0):
        krw = krw0 * s_W ** 2
        krow = (1 - s_W) ** 2
        return krw, krow

    def calc_mobility(self, kr):
        pass

    def get_delta_t(self, cfl):
        pass
