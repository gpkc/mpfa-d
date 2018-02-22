def xi(params):
    pass

def eta(params):
    pass

def A1(params):
    pass

def B4(params):
    pass

def C2(params): 
    pass

def D4(params):
    pass

def E3(params):
    pass

def F4(params):
    pass

def G4(params):
    pass

def H4(params):
    pass

def B1(params):
    pass

def D2(params):
    pass

def F3(params):
    pass

def LPEW2():
    for vertice in all_nodes:
        if vertice in dirichlet_nodes:
            pass
            # print('a dirichlet node')
        if vertice in neumann_nodes:
            pass
            # print('a neumann node')
        elif vertice in intern_nodes:
            volumes_around_vertice = mtu.get_bridge_adjacencies(vertice,
                                                                0, 3)
            volumes_around_vertex_tag = mb.tag_get_handle(
             "volumes around vertex {0}".format(vertice), 1,
             types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
            mb.tag_set_data(volumes_around_vertex_tag,
                            volumes_around_vertice,
                            np.repeat([1.0], len(volumes_around_vertice))
                            )
            for volume in volumes_around_vertice:
                faces_in_volume = set(mtu.get_bridge_adjacencies(
                                      volume, 3, 2))
                faces_sharing_vertice = set(mtu.get_bridge_adjacencies(
                                            vertice, 0, 2))
                faces_sharing_vertice = faces_in_volume.intersection(
                                        faces_sharing_vertice)
                adjacent_volumes = []
                for face in faces_sharing_vertice:
                    volumes_sharing_face = set(mtu.get_bridge_adjacencies(
                                            face, 2, 3))
                    side_volume = volumes_sharing_face - {volume}
                    adjacent_volumes.append(side_volume)
                M = volume
                W = adjacent_volumes.pop()

                #dado o vetor abaixo, esquerda e direita estão predeterminados

LPEW2()
