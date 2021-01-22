import pdb

import numpy as np
# from solvers.MpfaD import MpfaD3D
from solvers.nMpfaD import MpfaD3D
# from mpfad.interpolation.LPEW3 import LPEW3
from solvers.interpolation.LPEW3 import LPEW3
from preprocessor.mesh_preprocessor import MeshManager


class FlowChannel:

    def __init__(self):
        K_1 = np.array(
            [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
        )
        # mesh_test_conservative
        self.mesh = MeshManager("meshes/mesh_test_conservative.msh", dim=3)
        self.mesh.set_media_property("Permeability", {1: K_1}, dim_target=3)
        self.mesh.set_boundary_condition(
            "Dirichlet", {102: 1.0, 101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=False
        )
        self.mesh.get_redefine_centre()
        self.mesh.set_global_id()
        self.mpfad = MpfaD3D(self.mesh)
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)


class PiecewiseLinear:

    def __init__(self):
        K_1 = np.array(
            [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
        )
        K_2 = np.array(
            [0.5, 0.0, 0.0,
             0.0, 0.5, 0.0,
             0.0, 0.0, 0.5]
        )
        self.mesh = MeshManager(
            "meshes/geometry_two_regions_test.msh", dim=3
        )
        self.mesh.set_media_property(
            "Permeability", {1: K_1, 2: K_2}, dim_target=3
        )
        self.mesh.set_boundary_condition(
            "Dirichlet", {102: 1.0, 101: 0.0}, dim_target=2, set_nodes=True
        )
        self.mesh.set_boundary_condition(
            "Neumann", {201: 0.0}, dim_target=2, set_nodes=False
        )
        self.mesh.get_redefine_centre()
        self.mesh.set_global_id()
        self.mpfad = MpfaD3D(self.mesh)
        self.mpfad.run_solver(LPEW3(self.mesh).interpolate)
