import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util


class GenerateMesh:
    def __init__(self, mesh_file):

        self.mesh_file = mesh_file
        self.mb = core.Core()
        self.root_set = self.mb.get_root_set()
        self.mtu = topo_util.MeshTopoUtil(self.mb)

    def create_tags(self):

        self.gid_tag = self.mb.tag_get_handle(
            "GLOBAL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE, True
        )

        self.dirichlet_boundaries_tag = self.mb.tag_get_handle(
            "dirichlet_boundaries",
            1,
            types.MB_TYPE_INTEGER,
            types.MB_TAG_SPARSE,
            True,
        )

        self.material_set_tag = self.mb.tag_get_handle(
            "MATERIAL_SET", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True
        )

    def create_vert(self, vert_coords):
        vert = self.mb.create_vertices(vert_coords)
        return vert

    def create_volume(self, n_face, verts):
        if n_face == 4:
            mbtype = types.MBTET
        if n_face == 5:
            mbtype = types.MBPYRAMID
        if n_face == 6:
            mbtype = types.MBHEX
        if n_face > 6:
            mbtype = types.MBPRSIM
        return self.mb.create_element(mbtype, verts)

    def locate_information(self):
        with open(self.mesh_file) as msh:
            informations = []
            for i, line in enumerate(msh):
                line.strip().split("/n")
                if "Number of vertices" in line:
                    informations.append(i + 2)
                if "Number of control volume" in line:
                    informations.append(i + 2)
                if "Vertices  " in line:
                    informations.append(i + 2)
                if "Volumes->faces" in line:
                    informations.append(i + 2)
                if "Volumes->Verticess" in line:
                    informations.append(i + 2)
                if "Faces->Edgess" in line:
                    informations.append(i + 2)

        return informations

    def get_all_vertices(self):
        information_set = self.locate_information()
        first_vert = information_set[2] - 1
        last_vert = information_set[3] - 2
        msh = open(self.mesh_file)
        lines = msh.readlines()
        verts = lines[first_vert:last_vert]
        size = len(verts)
        verts = np.asarray([vert.split() for vert in verts]).reshape(size, 3)
        verts_coords = np.array(
            [[float(vert) for vert in verts[i]] for i in range(0, size)],
            dtype="float64",
        )
        msh.close()

        verts = self.mb.create_vertices(verts_coords.flatten())

        return verts

    def create_volumes(self):
        information_set = self.locate_information()
        first_volume = information_set[4] - 1
        last_volume = information_set[5] - 2
        msh = open(self.mesh_file)
        lines = msh.readlines()
        volumes = lines[first_volume:last_volume]
        volumes = np.asarray([volume.split() for volume in volumes])
        gid = 0
        vol_ms = self.mb.create_meshset()
        for i, volume in zip(range(0, len(volumes)), volumes):
            vol_verts = []
            for vert in volume[1:]:
                vol_verts.append(int(vert))
            el = self.create_volume(
                int(volume[0]), np.asarray(vol_verts, dtype="uint64")
            )
            self.mb.tag_set_data(self.gid_tag, el, gid)
            self.mb.add_entities(vol_ms, [el])
            gid += 1
        self.mb.tag_set_data(self.material_set_tag, el, 1)

        all_verts = self.mb.get_entities_by_dimension(0, 0)
        self.mtu.construct_aentities(all_verts)
        msh.close()

    def create_dirichlet_boundary_conditions(self):
        faces = self.mb.get_entities_by_dimension(0, 2)
        dirichlet_boundary_faces = self.mb.create_meshset()
        for face in faces:
            adjacent_vols = self.mtu.get_bridge_adjacencies(face, 2, 3)
            if len(adjacent_vols) < 2:
                self.mb.add_entities(dirichlet_boundary_faces, [face])
        self.mb.tag_set_data(
            self.material_set_tag, dirichlet_boundary_faces, 101
        )

    def write_msh_file(self, mesh_name):
        self.mb.write_file(mesh_name)
