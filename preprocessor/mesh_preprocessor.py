import numpy as np

from pymoab import core
from pymoab import types
from pymoab import topo_util


class MeshManager:
    def __init__(self, mesh_file, dim=3):

        self.dimension = dim
        self.mb = core.Core()
        self.root_set = self.mb.get_root_set()
        self.mtu = topo_util.MeshTopoUtil(self.mb)

        self.mb.load_file(mesh_file)

        self.physical_tag = self.mb.tag_get_handle("MATERIAL_SET")
        self.physical_sets = self.mb.get_entities_by_type_and_tag(
            0,
            types.MBENTITYSET,
            np.array((self.physical_tag,)),
            np.array((None,)),
        )

        # Initiate BC variables to the flux problem
        self.dirichlet_tag = self.mb.tag_get_handle(
            "Dirichlet", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        self.neumann_tag = self.mb.tag_get_handle(
            "Neumann", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        # Initiate BC and IC variables to the transport problem
        self.water_sat_tag = self.mb.tag_get_handle(
            "Water_Sat", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True,
        )

        self.water_sat_bc_tag = self.mb.tag_get_handle(
            "SW_BC", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True,
        )

        # Iniciate props to the transport problem
        self.oil_sat_i_tag = self.mb.tag_get_handle(
            "Oil_Sat_i", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True,
        )

        self.water_sat_i_tag = self.mb.tag_get_handle(
            "Water_Sat_i", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True,
        )

        self.rel_perm_w_tag = self.mb.tag_get_handle(
            "krW", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True,
        )

        self.rel_perm_o_tag = self.mb.tag_get_handle(
            "krO", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True,
        )

        self.face_mobility_tag = self.mb.tag_get_handle(
            "Mobility", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True,
        )
        self.node_pressure_tag = self.mb.tag_get_handle(
            "Node Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        # Iniciate material props
        self.perm_tag = self.mb.tag_get_handle(
            "Permeability", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        self.pressure_tag = self.mb.tag_get_handle(
            "Pressure", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        self.source_tag = self.mb.tag_get_handle(
            "Source term", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        self.volume_centre_tag = self.mb.tag_get_handle(
            "Volume centre", 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True
        )

        self.velocity_tag = self.mb.tag_get_handle(
            "Velocity", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True,
        )

        self.left_volume_tag = self.mb.tag_get_handle(
            "left volume", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True,
        )

        self.global_id_tag = self.mb.tag_get_handle(
            "Volume id", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE, True
        )

        self.all_volumes = self.mb.get_entities_by_dimension(0, self.dimension)

        self.all_nodes = self.mb.get_entities_by_dimension(0, 0)

        self.mtu.construct_aentities(self.all_nodes)
        self.all_faces = self.mb.get_entities_by_dimension(
            0, self.dimension - 1
        )

        self.dirichlet_faces = set()
        self.neumann_faces = set()
        self.sat_BC_faces = set()

    def create_vertices(self, coords):
        new_vertices = self.mb.create_vertices(coords)
        self.all_nodes.append(new_vertices)
        return new_vertices

    def create_element(self, poly_type, vertices):
        new_volume = self.mb.create_element(poly_type, vertices)
        self.all_volumes.append(new_volume)
        return new_volume

    def set_global_id(self):
        vol_ids = []
        range_of_ids = range(len(self.all_volumes))
        for id_, volume in zip(range_of_ids, self.all_volumes):
            vol_ids.append(id_)
        self.mb.tag_set_data(self.global_id_tag, self.all_volumes, vol_ids)

    def set_information(
        self, information_name, physicals_values, dim_target, set_connect=False
    ):
        try:
            information_tag = self.mb.tag_get_handle(information_name)

        except Exception:
            information_tag = self.mb.tag_get_handle(
                information_name,
                1,
                types.MB_TYPE_DOUBLE,
                types.MB_TAG_SPARSE,
                True,
            )

        for physical, value in physicals_values.items():
            for a_set in self.physical_sets:
                physical_group = self.mb.tag_get_data(
                    self.physical_tag, a_set, flat=True
                )

                if physical_group == physical:
                    group_elements = self.mb.get_entities_by_dimension(
                        a_set, dim_target
                    )

                    if information_name == "Dirichlet":
                        self.dirichlet_faces = self.dirichlet_faces | set(
                            group_elements
                        )

                    if information_name == "Neumann":
                        self.neumann_faces = self.neumann_faces | set(
                            group_elements
                        )

                    if information_name == "SW_BC":
                        self.sat_BC_faces = self.sat_BC_faces | set(
                            group_elements
                        )

                    for element in group_elements:
                        self.mb.tag_set_data(information_tag, element, value)

                        if set_connect:
                            connectivities = self.mtu.get_bridge_adjacencies(
                                element, 0, 0
                            )
                            self.mb.tag_set_data(
                                information_tag,
                                connectivities,
                                np.repeat(value, len(connectivities)),
                            )

    def set_media_property(
        self, property_name, physicals_values, dim_target=3, set_nodes=False
    ):
        self.set_information(
            property_name, physicals_values, dim_target, set_connect=set_nodes
        )

    def set_boundary_condition(
        self,
        boundary_condition,
        physicals_values,
        dim_target=3,
        set_nodes=False,
    ):
        self.set_information(
            boundary_condition,
            physicals_values,
            dim_target,
            set_connect=set_nodes,
        )

    def get_boundary_nodes(self):
        all_boundary_faces = self.dirichlet_faces | self.neumann_faces
        boundary_nodes = set()
        for face in all_boundary_faces:
            nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
            boundary_nodes.update(nodes)
        return boundary_nodes

    def intern_faces(self):
        return set(self.all_faces).difference(
            self.dirichlet_faces | self.neumann_faces
        )

    def get_non_boundary_volumes(self, dirichlet_nodes, neumann_nodes):
        volumes = self.all_volumes
        non_boundary_volumes = []
        for volume in volumes:
            volume_nodes = set(self.mtu.get_bridge_adjacencies(volume, 0, 0))
            if (
                volume_nodes.intersection(dirichlet_nodes | neumann_nodes)
            ) == set():
                non_boundary_volumes.append(volume)

        return non_boundary_volumes

    def get_bfaces_info(self):
        verts = []
        for face in self.dirichlet_faces:
            I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)
            left_volume = np.asarray(
                self.mtu.get_bridge_adjacencies(face, 2, 3), dtype="uint64"
            )

            JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
            JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
            LJ = (
                self.mb.get_coords([J])
                - self.mb.tag_get_data(self.volume_centre_tag, left_volume)[0]
            )
            N_IJK = np.cross(JI, JK) / 2.0
            test = np.dot(LJ, N_IJK)
            if test < 0.0:
                I, K = K, I
                JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
                JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
                N_IJK = np.cross(JI, JK) / 2.0
            verts.append(I, J, K)

    def get_redefine_centre(self):
        vol_centroids = []
        for volume in self.all_volumes:
            volume_centroid = self.mtu.get_average_position([volume])
            vol_centroids.append(volume_centroid)
        self.mb.tag_set_data(
            self.volume_centre_tag, self.all_volumes, vol_centroids
        )

    def _get_volumes_sharing_face_and_node(self, node, volume):
        vols_around_node = self.mtu.get_bridge_adjacencies(node, 0, 3)
        adj_vols = self.mtu.get_bridge_adjacencies(volume, 2, 3)
        volumes_sharing_face_and_node = set(adj_vols).difference(
            set(adj_vols).difference(set(vols_around_node))
        )
        return list(volumes_sharing_face_and_node)

    def _get_auxiliary_verts(self, node, volume, tao):
        # create meshset for volume and node
        pass

    def get_lpew2_support_region(self, tao):
        for node in self.all_nodes:
            self.volumes_around_node_ms = self.mb.create_meshset()
            volumes_around_node = self.mtu.get_bridge_adjacencies(node, 0, 3)
            self.mb.add_entities(
                self.volumes_around_node_ms, volumes_around_node
            )
            for volume in volumes_around_node:
                self.adj_volumes_ms = self.mb.create_meshset()
                self.aux_variables_ms = self.mb.create_meshset()
                adj_vols = self.mtu.get_bridge_adjacencies(volume, 2, 3)
                volumes_sharing_face_and_node = set(adj_vols).difference(
                    set(adj_vols).difference(set(volumes_around_node))
                )
                self.add_entities(
                    self.adj_volumes_ms, volumes_sharing_face_and_node
                )

    def get_node_cascade_lpew3(self, tao):
        for node in self.all_nodes:
            vols_around_node = self.mtu.get_bridge_adjacencies(node, 0, 3)
            for volume in vols_around_node:
                aux_verts = self.mtu.get_bridge_adjacencies(volume, 3, 0)
                aux_verts = list(set(aux_verts).difference({node}))
                adj_volumes = self.mtu.get_bridge_adjacencies(volume, 3, 3)
                adj_volumes = list(
                    set(adj_volumes).intersection({vols_around_node})
                )
                for adj_vol in adj_volumes:
                    adj_aux_verts = self.mtu.get_bridge_adjacencies(
                        adj_vol, 3, 0
                    )
                    aux_verts = list(set(adj_aux_verts).difference({node}))
                    adj_adj_volumes = self.mtu.get_bridge_adjacencies(
                        adj_vol, 3, 3
                    )
                    adj_adj_volumes = list(
                        set(adj_adj_volumes).intersection({vols_around_node})
                    )
                    aux_verts = [
                        tao * np.array(self.mb.get_coords[node])
                        + (1 - tao) * np.array(self.mb.get_coords(aux_vert))
                        for aux_vert in aux_verts
                    ]

    # TODO: This should calculate any generic polyhedral centroid
    def get_centroid(self, entity):
        verts = self.mb.get_adjacencies(entity, 0)
        coords = np.array([self.mb.get_coords([vert]) for vert in verts])
        qtd_pts = len(verts)
        coords = np.reshape(coords, (qtd_pts, 3))
        pseudo_cent = sum(coords) / qtd_pts
        return pseudo_cent

    # TODO: Should go on geometric
    def get_tetra_volume(self, tet_nodes):
        vect_1 = tet_nodes[1] - tet_nodes[0]
        vect_2 = tet_nodes[2] - tet_nodes[0]
        vect_3 = tet_nodes[3] - tet_nodes[0]
        vol_eval = abs(np.dot(np.cross(vect_1, vect_2), vect_3)) / 6.0
        return vol_eval
