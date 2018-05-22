import numpy as np

node_coords = np.asarray([.0, .0, .0])
nd_1_coords = np.asarray([-1., 0., 1.])
nd_2_coords = np.asarray([1., 0., 1.])
nd_3_coords = np.asarray([0., 1., 1.])
nd_4_coords = np.asarray([0., -1., 1.])
nd_5_coords = np.asarray([-1., 1., 1.])
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
adj_centroid = {}
adj_centroid[1] = [np.average([node_coords[i],
                               nd_1_coords[i],
                               nd_2_coords[i],
                               nd_4_coords[i]])for i in range(3)]
adj_centroid[2] = [np.average([node_coords[i],
                               nd_1_coords[i],
                               nd_2_coords[i],
                               nd_4_coords[i]])for i in range(3)]
adj_centroid[3] = [np.average([node_coords[i],
                               nd_2_coords[i],
                               nd_3_coords[i],
                               nd_5_coords[i]])for i in range(3)]
adj_centroid[4] = [np.average([node_coords[i],
                               nd_2_coords[i],
                               nd_3_coords[i],
                               nd_5_coords[i]])for i in range(3)]
adj_centroid[5] = [np.average([node_coords[i],
                               nd_3_coords[i],
                               nd_1_coords[i],
                               nd_6_coords[i]])for i in range(3)]
adj_centroid[6] = [np.average([node_coords[i],
                               nd_3_coords[i],
                               nd_6_coords[i],
                               nd_1_coords[i]])for i in range(3)]

faces = {}
faces[1] = [T[1], T[2]]
faces[2] = [T[2], T[3]]
faces[3] = [T[3], T[4]]
faces[4] = [T[4], T[5]]
faces[5] = [T[5], T[6]]
faces[6] = [T[6], T[1]]


def r_range(j, r):
    size = ((j - r + 5) % 6) + 1
    last_val = ((j + 4) % 6) + 1

    prod = [((last_val - k) % 6) + 1 for k in range(1, size+1)]
    return prod[::-1]


def _get_volume(node, face_verts, tetra_centroid):
    tetra_cords = np.append(face_verts, tetra_centroid).reshape([3, 3])
    tetra_vecs = [tetra_cords[j] - node for j in range(3)]
    tetra_vol = abs(np.linalg.det(tetra_vecs) / 6.)
    return tetra_vol

neta_ratio = []
for r in range(1, 7, 1):
    volume = _get_volume(node_coords, faces[r], tetra_c_coords)
    N_face = np.cross(T[r + 1], T[r]) / 2.

    N_i = np.cross(tetra_c_coords, T[r + 1]) / 2
    n_k = np.dot(N_i, np.dot(perm, N_face)) / (3. * volume)

    N_i = np.cross(T[r], tetra_c_coords) / 2
    n_k_plus = np.dot(N_i, np.dot(perm, N_face)) / (3. * volume)

    volume = _get_volume(node_coords, faces[r], adj_centroid[r])
    N_face_adj = -N_face

    N_i_adj = np.cross(T[r + 1], adj_centroid[r]) / 2
    n_k_adj = np.dot(N_i_adj, np.dot(perm, N_face_adj)) / (3. * volume)

    N_i_adj = np.cross(adj_centroid[r], T[r]) / 2
    n_k_adj_plus = np.dot(N_i_adj, np.dot(perm, N_face_adj)) / (3. * volume)
    neta_ratio.append(np.nan_to_num((n_k_adj - n_k)/ (n_k_adj_plus - n_k_plus)))
    print(n_k_adj, n_k, n_k_adj_plus, n_k_plus, neta_ratio)