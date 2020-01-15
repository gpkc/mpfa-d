from math import pi
from math import sqrt
import numpy as np

verts = [
    [1.0, 1.0, 1.0],
    [-1.25, -1.2, 1.10],
    [-1.0, 1.0, -1.0],
    [1.0, -1.0, -1.0],
]  # K


def benchmark_1(x, y, z):
    K = [1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0]
    u1 = 1 + np.sin(pi * x) * np.sin(pi * (y + 1 / 2)) * np.sin(
        pi * (z + 1 / 3)
    )
    return K, u1


def area_vector(JK, JI, test_vector=np.zeros(3)):
    N_IJK = np.cross(JK, JI) / 2.0
    if test_vector.all() != (np.zeros(3)).all():
        check_left_or_right = np.dot(test_vector, N_IJK)
        if check_left_or_right < 0:
            N_IJK = -N_IJK
    return N_IJK


def get_tetra_volume(tet_nodes):
    vect_1 = tet_nodes[1] - tet_nodes[0]
    vect_2 = tet_nodes[2] - tet_nodes[0]
    vect_3 = tet_nodes[3] - tet_nodes[0]
    vol_eval = abs(np.dot(np.cross(vect_1, vect_2), vect_3)) / 6.0
    return vol_eval


def _flux_term(vector_1st, permeab, vector_2nd, face_area):
    aux_1 = np.dot(vector_1st, permeab)
    aux_2 = np.dot(aux_1, vector_2nd)
    flux_term = aux_2 / face_area
    return flux_term


def _boundary_cross_term(
    tan_vector,
    norm_vector,
    face_area,
    tan_flux_term,
    norm_flux_term,
    cent_dist,
):
    mesh_aniso_term = np.dot(tan_vector, norm_vector) / (face_area ** 2)
    phys_aniso_term = tan_flux_term / face_area
    cross_term = (
        mesh_aniso_term * (norm_flux_term / cent_dist) - phys_aniso_term
    )
    return cross_term


centroid = np.asarray([0.0, 0.0, 0.0])
for axis in range(0, 3):
    for vert in verts:
        centroid[axis] += vert[axis] / 4

verts_solution = []
for vert in verts:
    verts_solution.append(benchmark_1(vert[0], vert[1], vert[2])[1])

u_c = benchmark_1(centroid[0], centroid[1], centroid[2])[1]

face_verts = [
    [verts[0], verts[1], verts[2]],
    [verts[0], verts[1], verts[3]],
    [verts[0], verts[2], verts[3]],
    [verts[1], verts[2], verts[3]],
]
perm = np.asarray(benchmark_1(0.0, 0.0, 0.0)[0])

flux = []
all_N_IJK = []

for face in face_verts:
    I, J, K = np.asarray(face)
    g_I = np.asarray(benchmark_1(I[0], I[1], I[2])[1])
    g_J = np.asarray(benchmark_1(J[0], J[1], J[2])[1])
    g_K = np.asarray(benchmark_1(K[0], K[1], K[2])[1])
    JI = I - J
    JK = K - J
    face_centroid = (I + J + K) / 3
    test_vector = face_centroid - centroid
    N_IJK = area_vector(JK, JI, test_vector)
    all_N_IJK.append(N_IJK)

    # print(N_IJK)
    face_area = np.sqrt(np.dot(N_IJK, N_IJK))
    JR = np.asarray(centroid) - np.asarray(J)
    h_R = np.absolute(np.dot(N_IJK, JR) / np.sqrt(np.dot(N_IJK, N_IJK)))
    tan_JI = np.cross(JI, N_IJK)
    if np.dot(tan_JI, JK) < 0:
        print("true")
        tan_JI = -tan_JI

    tan_JK = np.cross(N_IJK, JK)
    if np.dot(tan_JK, JI) < 0:
        print("also true")
        tan_JK = -tan_JK

    K_R = np.asarray(perm).reshape([3, 3])
    K_R_n = _flux_term(N_IJK, K_R, N_IJK, face_area)
    K_R_JI = _flux_term(N_IJK, K_R, tan_JI, face_area)
    K_R_JK = _flux_term(N_IJK, K_R, tan_JK, face_area)

    D_JI = _boundary_cross_term(tan_JK, JR, face_area, K_R_JK, K_R_n, h_R)
    D_JK = _boundary_cross_term(tan_JI, JR, face_area, K_R_JI, K_R_n, h_R)

    K_n_eff = K_R_n / h_R
    flux.append(
        -(2 * K_n_eff * (u_c - g_J) + D_JI * (g_J - g_I) + D_JK * (g_J - g_K))
    )

print(flux, sum(flux), verts_solution, u_c)
