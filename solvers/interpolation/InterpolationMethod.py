import numpy as np

from pymoab import types


class InterpolationMethodBase:
    def __init__(self, mesh_data):
        self.mesh_data = mesh_data
        self.mb = mesh_data.mb
        self.mtu = mesh_data.mtu
        self.dirichlet_tag = mesh_data.dirichlet_tag
        self.neumann_tag = mesh_data.neumann_tag
        self.perm_tag = mesh_data.perm_tag
        # self.node_wts_tag = mesh_data.node_wts_tag

        self._set_nodes(mesh_data)
        self._set_faces(mesh_data)

    def _set_nodes(self, mesh_data):
        self.dirichlet_nodes = set(
            self.mb.get_entities_by_type_and_tag(
                0, types.MBVERTEX, self.dirichlet_tag, np.array((None,))
            )
        )

        self.neumann_nodes = set(
            self.mb.get_entities_by_type_and_tag(
                0, types.MBVERTEX, self.neumann_tag, np.array((None,))
            )
        )
        self.neumann_nodes = self.neumann_nodes - self.dirichlet_nodes

        boundary_nodes = self.dirichlet_nodes | self.neumann_nodes
        self.intern_nodes = set(mesh_data.all_nodes) - boundary_nodes

    def _set_faces(self, mesh_data):
        self.dirichlet_faces = mesh_data.dirichlet_faces
        self.neumann_faces = mesh_data.neumann_faces

        self.all_faces = self.mb.get_entities_by_dimension(0, 2)
        boundary_faces = self.dirichlet_faces | self.neumann_faces

        self.intern_faces = set(self.all_faces) - boundary_faces

    def interpolate(self, node):
        raise NotImplementedError
