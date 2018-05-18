import numpy as np

node_coords = np.asarray([.5, .5, .5])
nd_1_coords = np.asarray([1., 0., 0.])
nd_2_coords = np.asarray([0., 1., 0.])
nd_3_coords = np.asarray([0., 0., 1.])
nd_4_coords = np.asarray([0., 1., 1.])
nd_5_coords = np.asarray([1., 1., 0.])
nd_6_coords = np.asarray([1., 1., 1.])

perm = [[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]]

big_tetra_vol = abs(np.linalg.det([nd_1_coords - node_coords,
                                   nd_2_coords - node_coords,
                                   nd_3_coords - node_coords]) / 6.)
T = {}

T[1] = [np.average([node_coords[i], nd_1_coords[i]]) for i in range(3)]
T[2] = [np.average([node_coords[i], nd_1_coords[i],
                    nd_2_coords[i]]) for i in range(3)]
T[3] = [np.average([node_coords[i], nd_2_coords[i]]) for i in range(3)]
T[4] = [np.average([node_coords[i], nd_2_coords[i],
                    nd_3_coords[i]])for i in range(3)]
T[5] = [np.average([node_coords[i], nd_3_coords[i]]) for i in range(3)]
T[6] = [np.average([node_coords[i], nd_3_coords[i],
                    nd_1_coords[i]])for i in range(3)]
T[7] = T[1]
tetra_c_coords = [np.average([node_coords[i],
                                     nd_1_coords[i],
                                     nd_2_coords[i],
                                     nd_3_coords[i]])for i in range(3)]
adj_coords = {}
adj_coords[1] = [np.average([node_coords[i],
                                     nd_1_coords[i],
                                     nd_2_coords[i],
                                     nd_4_coords[i]])for i in range(3)]
adj_coords[2] = [np.average([node_coords[i],
                                     nd_1_coords[i],
                                     nd_2_coords[i],
                                     nd_4_coords[i]])for i in range(3)]
adj_coords[3] = [np.average([node_coords[i],
                                     nd_1_coords[i],
                                     nd_2_coords[i],
                                     nd_5_coords[i]])for i in range(3)]
adj_coords[4] = [np.average([node_coords[i],
                                     nd_1_coords[i],
                                     nd_2_coords[i],
                                     nd_5_coords[i]])for i in range(3)]
adj_coords[5] = [np.average([node_coords[i],
                                     nd_1_coords[i],
                                     nd_2_coords[i],
                                     nd_6_coords[i]])for i in range(3)]
adj_coords[6] = [np.average([node_coords[i],
                                     nd_1_coords[i],
                                     nd_2_coords[i],
                                     nd_6_coords[i]])for i in range(3)]

faces = {}
faces[1] = [node_coords, T[1], T[2]]
faces[2] = [node_coords, T[2], T[3]]
faces[3] = [node_coords, T[3], T[4]]
faces[4] = [node_coords, T[4], T[5]]
faces[5] = [node_coords, T[5], T[6]]
faces[6] = [node_coords, T[6], T[1]]


def get_neta(T_r, T_r_plus_1, face_r, tetra_centroid):
    AB = T_r - node_coords
    CB = T_r_plus_1 - node_coords
    DB = tetra_centroid - node_coords
    tetra_cords = np.append(face_r, tetra_c_coords).reshape([4, 3])
    tetra_vecs = [tetra_cords[j] - node_coords for j in range(1, 4)]
    tetra_vol = abs(np.linalg.det(tetra_vecs) / 6.)
    N_face = np.cross(CB, AB) / 2.
    N_i = np.cross(DB, CB) / 2.
    neta = np.dot(np.dot(N_i, perm), N_face) / (3. * tetra_vol)
    return neta


for r in range(1, 7, 1):
    print(get_neta(T[r], T[r + 1], faces[r], tetra_c_coords),
          get_neta(T[r], T[r + 1], faces[r], adj_coords[r]))
