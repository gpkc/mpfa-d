import numpy as np
import pdb

np.seterr(all="ignore")
node_coords = np.asarray([0.0, 0.0, 0.0])
nd_1_coords = np.asarray([-1.0, 0.0, 1.0])
nd_2_coords = np.asarray([1.0, 0.0, 1.0])
nd_3_coords = np.asarray([0.0, 1.0, 1.0])
nd_4_coords = np.asarray([0.0, -1.0, 1.0])
nd_5_coords = np.asarray([-1.0, 1.0, 1.0])
nd_6_coords = np.asarray([1.0, 1.0, 1.0])

perm = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

adj_perm = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

big_tetra_vol = abs(
    np.linalg.det(
        [
            nd_1_coords - node_coords,
            nd_2_coords - node_coords,
            nd_3_coords - node_coords,
        ]
    )
    / 6.0
)
T = {}

T[1] = [np.average([node_coords[i], nd_1_coords[i]]) for i in range(3)]
T[2] = [
    np.average([node_coords[i], nd_1_coords[i], nd_2_coords[i]])
    for i in range(3)
]
T[3] = [np.average([node_coords[i], nd_2_coords[i]]) for i in range(3)]
T[4] = [
    np.average([node_coords[i], nd_2_coords[i], nd_3_coords[i]])
    for i in range(3)
]
T[5] = [np.average([node_coords[i], nd_3_coords[i]]) for i in range(3)]
T[6] = [
    np.average([node_coords[i], nd_3_coords[i], nd_1_coords[i]])
    for i in range(3)
]
T[7] = T[1]

tetra_c_coords = [
    np.average(
        [node_coords[i], nd_1_coords[i], nd_2_coords[i], nd_3_coords[i]]
    )
    for i in range(3)
]
adj_centroid = {}
adj_centroid[1] = [
    np.average(
        [node_coords[i], nd_1_coords[i], nd_2_coords[i], nd_4_coords[i]]
    )
    for i in range(3)
]
adj_centroid[2] = [
    np.average(
        [node_coords[i], nd_1_coords[i], nd_2_coords[i], nd_4_coords[i]]
    )
    for i in range(3)
]
adj_centroid[3] = [
    np.average(
        [node_coords[i], nd_1_coords[i], nd_3_coords[i], nd_5_coords[i]]
    )
    for i in range(3)
]
adj_centroid[4] = [
    np.average(
        [node_coords[i], nd_1_coords[i], nd_3_coords[i], nd_5_coords[i]]
    )
    for i in range(3)
]
adj_centroid[5] = [
    np.average(
        [node_coords[i], nd_3_coords[i], nd_2_coords[i], nd_6_coords[i]]
    )
    for i in range(3)
]
adj_centroid[6] = [
    np.average(
        [node_coords[i], nd_3_coords[i], nd_6_coords[i], nd_2_coords[i]]
    )
    for i in range(3)
]

faces = {}
faces[1] = [T[1], T[2]]
faces[2] = [T[2], T[3]]
faces[3] = [T[3], T[4]]
faces[4] = [T[4], T[5]]
faces[5] = [T[5], T[6]]
faces[6] = [T[6], T[1]]

# print(tetra_c_coords, T[2], adj_centroid[1])
Q = node_coords
vec_1 = np.cross(tetra_c_coords - Q, T[2] - Q)
vec_2 = np.cross(T[2] - Q, adj_centroid[2] - Q)
# print(np.linalg.det([tetra_c_coords, Q, adj_centroid[1]]))


def r_range(j, r):
    size = ((j - r + 5) % 6) + 1
    last_val = ((j + 4) % 6) + 1

    prod = [((last_val - k) % 6) + 1 for k in range(1, size + 1)]
    return prod[::-1]


def _get_volume(node, face_verts, tetra_centroid):
    tetra_cords = np.append(face_verts, tetra_centroid).reshape([3, 3])
    tetra_vecs = [tetra_cords[j] - node for j in range(3)]
    tetra_vol = abs(np.linalg.det(tetra_vecs) / 6.0)
    return tetra_vol


for j in range(1, 7, 1):
    for r in range(1, 7, 1):
        neta_ratio = []
        prod = 1.0
        for r_star in r_range(j, r):
            volume = _get_volume(node_coords, faces[r], tetra_c_coords)
            N_face = np.cross(T[r_star + 1], T[r_star]) / 2.0

            N_i = np.cross(tetra_c_coords, T[r + 1]) / 2
            n_k = np.dot(N_i, np.dot(perm, N_face)) / (3.0 * volume)

            N_i = np.cross(T[r_star], tetra_c_coords) / 2
            n_k_plus = np.dot(N_i, np.dot(perm, N_face)) / (3.0 * volume)

            volume = _get_volume(
                node_coords, faces[r_star], adj_centroid[r_star]
            )
            N_face_adj = N_face

            N_i_adj = np.cross(T[r_star + 1], adj_centroid[r_star]) / 2
            n_k_adj = np.dot(N_i_adj, np.dot(adj_perm, N_face_adj)) / (
                3.0 * volume
            )

            N_i_adj = np.cross(adj_centroid[r_star], T[r_star]) / 2
            n_k_adj_plus = np.dot(N_i_adj, np.dot(adj_perm, N_face_adj)) / (
                3.0 * volume
            )
            if (n_k_adj_plus - n_k_plus) == 0:
                neta = 0.0
                neta_ratio.append(
                    (np.nan_to_num(n_k_adj - n_k) / (n_k_adj_plus - n_k_plus))
                )
            else:
                neta = (n_k_adj - n_k) / (n_k_adj_plus - n_k_plus)
                neta_ratio.append(
                    (np.nan_to_num(n_k_adj - n_k) / (n_k_adj_plus - n_k_plus))
                )

            prod = (-1) ** (1 + r_star) * neta * prod
            # print(j, r, r_star, n_k_adj, n_k, n_k_adj_plus, n_k_plus, neta)
            # print(r_star, n_k_adj, n_k, n_k_adj_plus, n_k_plus, neta)
        # print(j, r, r_range(j, r), neta_ratio, prod)
