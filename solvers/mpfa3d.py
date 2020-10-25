import numpy as np
from tpfa import tpfaScheme
from mdot import mdot
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class mpfa3dScheme(tpfaScheme):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.w, self.fw = self.weights()
        (
            self.dji,
            self.djk,
            self.keq,
            self.dirichletT,
            self.dirichletQ,
            self.neumann,
        ) = self.preparacao()
        self.T, self.Q = self.assembly_mpfa3d()
        self.p = spsolve(self.T, self.Q)

    def assembly_mpfa3d(self):

        noI = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.internal, 0, 0
        )[:, 0]
        noJ = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.internal, 0, 0
        )[:, 1]
        noK = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.internal, 0, 0
        )[:, 2]

        area, N = self.calc_area(self.mesh.faces.internal)
        veri = self.calc_veri(self.mesh.faces.internal, N)
        left = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.internal, 2, 3
        )[:, 0]
        right = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.internal, 2, 3
        )[:, 1]
        left[veri], right[veri] = right[veri], left[veri]

        gi = 0.5 * self.djk
        gj = 0.5 * self.dji - 0.5 * self.djk
        gk = -0.5 * self.dji

        Gijk = (
            self.keq * gi * self.node_value[noI]
            + self.keq * gj * self.node_value[noJ]
            + self.keq * gk * self.node_value[noK]
        )

        Fijk = (
            self.keq * gi * self.fw[noI]
            + self.keq * gj * self.fw[noJ]
            + self.keq * gk * self.fw[noK]
        )

        rowT = np.concatenate([left, left, right, right])
        colT = np.concatenate([left, right, right, left])
        dataT = np.concatenate([self.keq, -self.keq, self.keq, -self.keq])

        rowQ = np.concatenate([left, right, left, right])
        colQ = np.zeros(2 * len(left) + 2 * len(right))
        dataQ = np.concatenate([Gijk, -Gijk, Fijk, -Fijk])

        esurn = self.mesh.nodes.bridge_adjacencies(self.mesh.nodes.all, 0, 3)

        for i in range(len(self.mesh.faces.internal)):
            rowT = np.append(
                rowT,
                np.concatenate(
                    [
                        left[i] * np.ones(len(esurn[noI[i]])),
                        right[i] * np.ones(len(esurn[noI[i]])),
                        left[i] * np.ones(len(esurn[noJ[i]])),
                        right[i] * np.ones(len(esurn[noJ[i]])),
                        left[i] * np.ones(len(esurn[noK[i]])),
                        right[i] * np.ones(len(esurn[noK[i]])),
                    ]
                ),
            )
            colT = np.append(
                colT,
                np.concatenate(
                    [
                        esurn[noI[i]],
                        esurn[noI[i]],
                        esurn[noJ[i]],
                        esurn[noJ[i]],
                        esurn[noK[i]],
                        esurn[noK[i]],
                    ]
                ),
            )
            dataT = np.append(
                dataT,
                np.concatenate(
                    [
                        -self.keq[i] * gi[i] * self.w[noI[i]],
                        self.keq[i] * gi[i] * self.w[noI[i]],
                        -self.keq[i] * gj[i] * self.w[noJ[i]],
                        self.keq[i] * gj[i] * self.w[noJ[i]],
                        -self.keq[i] * gk[i] * self.w[noK[i]],
                        self.keq[i] * gk[i] * self.w[noK[i]],
                    ]
                ),
            )

        leftb = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.boundary, 2, 3
        )[:, 0]
        rowT = np.append(rowT, leftb)
        colT = np.append(colT, leftb)
        dataT = np.append(dataT, self.dirichletT)
        rowQ = np.append(rowQ, [leftb, leftb])
        colQ = np.append(colQ, [np.zeros(len(leftb)), np.zeros(len(leftb))])
        dataQ = np.append(dataQ, [-self.dirichletQ, -self.neumann])

        T = csc_matrix(
            (dataT, (rowT, colT)),
            shape=(len(self.mesh.volumes), len(self.mesh.volumes)),
        )

        Q = csc_matrix(
            (dataQ, (rowQ, colQ)), shape=(len(self.mesh.volumes), 1)
        )

        return T, Q

    def preparacao(self):

        permeabilities = self.vol_permeabilities()

        # PARA FACES INTERNAS: ----------------------------------------------------------------------------------------
        area, N = self.calc_area(self.mesh.faces.internal)
        veri = self.calc_veri(self.mesh.faces.internal, N)

        L = self.mesh.volumes.center(
            self.mesh.faces.bridge_adjacencies(self.mesh.faces.internal, 2, 3)[
                :, 0
            ]
        )
        R = self.mesh.volumes.center(
            self.mesh.faces.bridge_adjacencies(self.mesh.faces.internal, 2, 3)[
                :, 1
            ]
        )
        L[veri], R[veri] = R[veri], L[veri]

        left = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.internal, 2, 3
        )[:, 0]
        right = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.internal, 2, 3
        )[:, 1]
        left[veri], right[veri] = right[veri], left[veri]

        heights = self.calc_heights(self.mesh.faces.internal, N, L, R, 2)
        hL = heights[0]
        hR = heights[1]

        KL = permeabilities[left]
        KR = permeabilities[right]

        keq, dji, djk = self.calc_keq_dji_djk(N, L, R, KL, KR, hL, hR, area)
        # FIM DO TRECHO PARA FACES INTERNAS ---------------------------------------------------------------------------

        # PARA FACES DE CONTORNO: -------------------------------------------------------------------------------------
        area, N = self.calc_area(self.mesh.faces.boundary)
        veri = self.calc_veri(self.mesh.faces.boundary, N)

        N[veri] = -N[veri]
        L = self.mesh.volumes.center(
            self.mesh.faces.bridge_adjacencies(self.mesh.faces.boundary, 2, 3)[
                :, 0
            ]
        )

        hL = self.calc_heights(self.mesh.faces.boundary, N, L, R, 1)

        KL = permeabilities[
            self.mesh.faces.bridge_adjacencies(self.mesh.faces.boundary, 2, 3)[
                :, 0
            ]
        ]

        dirichletT, dirichletQ, neumann = self.calc_boundary_contribution(
            N, L, KL, hL, area
        )
        # FIM DO TRECHO PARA FACES DE CONTORNO: -----------------------------------------------------------------------

        return dji, djk, keq, dirichletT, dirichletQ, neumann

    def vol_permeabilities(self):

        region_flag = np.zeros(len(self.mesh.volumes))
        for key in self.mesh.volumes.flag:
            for i in self.mesh.volumes.flag[key]:
                region_flag[i] = key

        permeabilities = []
        for i in range(len(region_flag)):
            permeabilities.append(self.permeabilities[region_flag[i]])
        permeabilities = np.array(permeabilities)

        return permeabilities

    def calc_area(self, faces):

        I = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(faces, 0, 0)[:, 0]
        )
        J = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(faces, 0, 0)[:, 1]
        )
        K = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(faces, 0, 0)[:, 2]
        )

        N = 0.5 * np.cross(I - J, K - J)
        area = np.linalg.norm(N, axis=1)

        return area, N

    def calc_veri(self, faces, N):

        tdot = lambda x, y: (x * y).sum(axis=1)

        L = self.mesh.volumes.center(
            self.mesh.faces.bridge_adjacencies(faces, 2, 3)[:, 0]
        )
        I = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(faces, 0, 0)[:, 0]
        )

        veri = np.nonzero(
            (tdot(I - L, N) - np.absolute(tdot(I - L, N)))
            / np.absolute(tdot(I - L, N))
        )

        return veri

    def calc_heights(self, faces, N, L, R, i):

        tdot = lambda x, y: (x * y).sum(axis=1)

        normal = np.transpose(N.T / np.linalg.norm(N, axis=1))
        I = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(faces, 0, 0)[:, 0]
        )

        if i == 2:
            heights = np.array(
                [
                    np.absolute(tdot(I - L, normal)),
                    np.absolute(tdot(I - R, normal)),
                ]
            )
        else:
            heights = np.array(np.absolute(tdot(I - L, normal)))

        return heights

    def calc_keq_dji_djk(self, N, L, R, KL, KR, hL, hR, area):

        tdot = lambda x, y: (x * y).sum(axis=1)

        I = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(self.mesh.faces.internal, 0, 0)[
                :, 0
            ]
        )
        J = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(self.mesh.faces.internal, 0, 0)[
                :, 1
            ]
        )
        K = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(self.mesh.faces.internal, 0, 0)[
                :, 2
            ]
        )

        JI = I - J
        JK = K - J
        LR = R - L
        TJI = np.cross(N, JI)
        TJK = np.cross(N, JK)

        KnL = mdot(N, KL, N) / (area * area)
        KnR = mdot(N, KR, N) / (area * area)

        keq = (area * KnL * KnR) / (hL * KnR + hR * KnL)

        KTJIL = mdot(N, KL, TJI) / (area * area)
        KTJKR = mdot(N, KR, TJK) / (area * area)
        KTJIR = mdot(N, KR, TJI) / (area * area)
        KTJKL = mdot(N, KL, TJK) / (area * area)

        dji = (tdot(TJI, LR) / (area * area)) - (1 / area) * (
            hL * (KTJIL / KnL) + hR * (KTJIR / KnR)
        )
        djk = (tdot(TJK, LR) / (area * area)) - (1 / area) * (
            hL * (KTJKL / KnL) + hR * (KTJKR / KnR)
        )

        return keq, dji, djk

    def calc_boundary_contribution(self, N, L, KL, hL, area):

        tdot = lambda x, y: (x * y).sum(axis=1)

        noI = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.boundary, 0, 0
        )[:, 0]
        noJ = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.boundary, 0, 0
        )[:, 1]
        noK = self.mesh.faces.bridge_adjacencies(
            self.mesh.faces.boundary, 0, 0
        )[:, 2]

        I = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(self.mesh.faces.boundary, 0, 0)[
                :, 0
            ]
        )
        J = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(self.mesh.faces.boundary, 0, 0)[
                :, 1
            ]
        )
        K = self.mesh.nodes.coords(
            self.mesh.faces.bridge_adjacencies(self.mesh.faces.boundary, 0, 0)[
                :, 2
            ]
        )

        JI = I - J
        JK = K - J
        LJ = J - L

        veri = np.nonzero(
            (
                tdot(np.cross(JI, JK), N)
                - np.absolute(tdot(np.cross(JI, JK), N))
            )
            / np.absolute(tdot(np.cross(JI, JK), N))
        )

        I[veri], K[veri] = K[veri], I[veri]
        noI[veri], noK[veri] = noK[veri], noI[veri]

        JI = I - J
        JK = K - J

        TJI = np.cross(N, JI)
        TJK = np.cross(N, JK)

        pI = self.node_value[noI]
        pJ = self.node_value[noJ]
        pK = self.node_value[noK]

        KnL = mdot(N, KL, N) / (area * area)
        KTJIL = mdot(N, KL, TJI) / (area * area)
        KTJKL = mdot(N, KL, TJK) / (area * area)

        dirichletQ = (1 / (2 * hL * area)) * (
            (tdot(-TJK, LJ) * KnL + hL * area * KTJKL) * (pI - pJ)
            - 2 * (area * area) * KnL * pJ
            + (tdot(-TJI, LJ) * KnL + hL * area * KTJIL) * (pJ - pK)
        )

        dirichletT = (1 / (2 * hL * area)) * (2 * (area * area) * KnL)

        neumann = np.zeros(len(self.mesh.faces.boundary))
        for key in self.mesh.faces.flag:
            for i in self.mesh.faces.flag[key]:
                neumann[i] = area[i] * self.flag_value[key] * (key > 200)
                dirichletQ[i] *= key < 200
                dirichletT[i] *= key < 200

        return dirichletT, dirichletQ, neumann

    def weights(self):

        esurn = self.mesh.nodes.bridge_adjacencies(self.mesh.nodes.all, 0, 3)

        w = []
        fw = []
        node = 0

        for i in esurn:

            Q = self.mesh.nodes.coords(node)
            noI, noJ, noK, ocI, ocJ, ocK, ofI, ofJ, ofK = self.calc_nos_viz(
                i, node
            )

            I = self.mesh.nodes.coords(noI)
            J = self.mesh.nodes.coords(noJ)
            K = self.mesh.nodes.coords(noK)

            T = np.array(
                [
                    0.5 * (Q + I),
                    (1 / 3) * (Q + I + K),
                    0.5 * (Q + K),
                    (1 / 3) * (Q + J + K),
                    0.5 * (Q + J),
                    (1 / 3) * (Q + I + J),
                ]
            )
            t = np.array([[0, 5, 1], [1, 3, 2], [3, 5, 4], [1, 5, 3]])

            csi = self.calc_csi(Q, T, t, i)

            neta, sigma, neubf = self.calc_neta_sigma(
                node, Q, T, ocI, ocJ, ocK, ofI, ofJ, ofK, i
            )

            lam, gam = self.calc_lam(
                node, csi, neta, sigma, neubf, ocI, ocJ, ocK, i
            )

            S = sum(lam) if any(lam != 0) else 1

            w.append(lam / S)

            fw.append(gam / S)

            node = node + 1

        w = np.array(w)
        fw = np.array(fw)

        return w, fw

    def calc_nos_viz(self, esurn, node):

        tdot = lambda x, y: (x * y).sum(axis=1)

        othernodes = self.mesh.volumes.bridge_adjacencies(esurn, 3, 0)[
            self.mesh.volumes.bridge_adjacencies(esurn, 3, 0) != node
        ].reshape((len(esurn), 3))

        elemfaceelem = self.mesh.volumes.bridge_adjacencies(esurn, 2, 3)

        noI = othernodes[:, 0]
        noJ = othernodes[:, 1]
        noK = othernodes[:, 2]

        I = self.mesh.nodes.coords(noI)
        J = self.mesh.nodes.coords(noJ)
        K = self.mesh.nodes.coords(noK)

        N = 0.5 * np.cross(I - J, K - J)
        C = self.mesh.volumes.center(esurn)

        veri = np.nonzero(
            (tdot(I - C, N) - np.absolute(tdot(I - C, N)))
            / np.absolute(tdot(I - C, N))
        )

        noJ[veri], noK[veri] = noK[veri], noJ[veri]

        elemsurnoI = self.mesh.nodes.bridge_adjacencies(noI, 0, 3)
        elemsurnoJ = self.mesh.nodes.bridge_adjacencies(noJ, 0, 3)
        elemsurnoK = self.mesh.nodes.bridge_adjacencies(noK, 0, 3)
        ocI = -1 * np.ones(len(elemsurnoI))
        ocJ = -1 * np.ones(len(elemsurnoJ))
        ocK = -1 * np.ones(len(elemsurnoK))

        for j in range(len(elemsurnoI)):
            for e in elemfaceelem[j][:]:
                if (
                    len([d for d, x in enumerate(e == elemsurnoI[j][:]) if x])
                    == 0
                ):
                    ocI[j] = e

        for j in range(len(elemsurnoJ)):
            for e in elemfaceelem[j][:]:
                if (
                    len([d for d, x in enumerate(e == elemsurnoJ[j][:]) if x])
                    == 0
                ):
                    ocJ[j] = e

        for j in range(len(elemsurnoK)):
            for e in elemfaceelem[j][:]:
                if (
                    len([d for d, x in enumerate(e == elemsurnoK[j][:]) if x])
                    == 0
                ):
                    ocK[j] = e

        faceelem = self.mesh.volumes.bridge_adjacencies(esurn, 2, 2)

        facesurnoI = self.mesh.nodes.bridge_adjacencies(noI, 0, 2)
        facesurnoJ = self.mesh.nodes.bridge_adjacencies(noJ, 0, 2)
        facesurnoK = self.mesh.nodes.bridge_adjacencies(noK, 0, 2)
        ofI = -1 * np.ones(len(facesurnoI))
        ofJ = -1 * np.ones(len(facesurnoJ))
        ofK = -1 * np.ones(len(facesurnoK))

        for j in range(len(facesurnoI)):
            for e in faceelem[j][:]:
                if (
                    len([d for d, x in enumerate(e == facesurnoI[j][:]) if x])
                    == 0
                ):
                    ofI[j] = e

        for j in range(len(facesurnoJ)):
            for e in faceelem[j][:]:
                if (
                    len([d for d, x in enumerate(e == facesurnoJ[j][:]) if x])
                    == 0
                ):
                    ofJ[j] = e

        for j in range(len(facesurnoK)):
            for e in faceelem[j][:]:
                if (
                    len([d for d, x in enumerate(e == facesurnoK[j][:]) if x])
                    == 0
                ):
                    ofK[j] = e

        return noI, noJ, noK, ocI, ocJ, ocK, ofI, ofJ, ofK

    def calc_csi(self, Q, T, t, esurn):

        tdot = lambda x, y: (x * y).sum(axis=1)

        permeabilities = self.vol_permeabilities()
        Kk = permeabilities[esurn]

        NQ = 0.5 * np.cross(
            T[t[:, 0], :, :] - T[t[:, 1], :, :],
            T[t[:, 2], :, :] - T[t[:, 1], :, :],
        )
        QI = T[t[:, 0], :, :] - Q
        QJ = T[t[:, 1], :, :] - Q
        QK = T[t[:, 2], :, :] - Q
        Nti = 0.5 * np.cross(QJ, QK)
        Ntj = 0.5 * np.cross(QK, QI)
        Ntk = 0.5 * np.cross(QI, QJ)
        csi = []

        for i in range(len(esurn)):
            csi.append([0.0] * 6)

        csi = np.array(csi)

        for i in range(len(NQ)):
            vol_t = abs((1 / 6) * tdot(2 * Ntk[i], QK[i]))
            csi[:, t[i, 0]] += mdot(NQ[i], Kk, Nti[i]) / (3 * vol_t)
            csi[:, t[i, 1]] += mdot(NQ[i], Kk, Ntj[i]) / (3 * vol_t)
            csi[:, t[i, 2]] += mdot(NQ[i], Kk, Ntk[i]) / (3 * vol_t)

        csi = csi.T

        return csi

    def calc_neta_sigma(self, node, Q, T, ocI, ocJ, ocK, ofI, ofJ, ofK, esurn):

        tdot = lambda x, y: (x * y).sum(axis=1)

        permeabilities = self.vol_permeabilities()
        Kk = permeabilities[esurn]
        eps = 1e-9
        KkoI = []
        KkoJ = []
        KkoK = []
        CoI = []
        CoJ = []
        CoK = []
        KK = np.array([[0.0] * len(esurn) for y in range(6)])
        area = np.zeros_like(KK)
        height = np.zeros_like(KK)
        Nf = np.zeros_like(T)
        t = list(range(6))
        u = [1, 2, 3, 4, 5, 0]
        C = self.mesh.volumes.center(esurn)
        C += eps * np.random.rand(*C.shape)
        valueopI = []
        valueopJ = []
        valueopK = []

        for i in range(len(esurn)):
            KkoI.append(permeabilities[int(ocI[i])]) if ocI[
                i
            ] >= 0 else KkoI.append([[0] * 3 for y in range(3)])
            KkoJ.append(permeabilities[int(ocJ[i])]) if ocJ[
                i
            ] >= 0 else KkoJ.append([[0] * 3 for y in range(3)])
            KkoK.append(permeabilities[int(ocK[i])]) if ocK[
                i
            ] >= 0 else KkoK.append([[0] * 3 for y in range(3)])
            CoI.append(self.mesh.volumes.center(int(ocI[i]))) if ocI[
                i
            ] >= 0 else CoI.append(Q)
            CoJ.append(self.mesh.volumes.center(int(ocJ[i]))) if ocJ[
                i
            ] >= 0 else CoJ.append(Q)
            CoK.append(self.mesh.volumes.center(int(ocK[i]))) if ocK[
                i
            ] >= 0 else CoK.append(Q)
            valueopI.append(self.flag_value[self.face_flag[int(ofI[i])]])
            valueopJ.append(self.flag_value[self.face_flag[int(ofJ[i])]])
            valueopK.append(self.flag_value[self.face_flag[int(ofK[i])]])

        KkoI = np.array(KkoI)
        KkoJ = np.array(KkoJ)
        KkoK = np.array(KkoK)
        CoI = np.array(CoI)
        CoJ = np.array(CoJ)
        CoK = np.array(CoK)

        for i in range(6):
            Nf[i] = 0.5 * np.cross(T[u[i]] - Q, T[t[i]] - Q)
            area[i] = np.linalg.norm(Nf[i], axis=1)
            height[i] = tdot(T[t[i]] - C, Nf[i]) / np.linalg.norm(
                Nf[i], axis=1
            )

        hoI = eps + tdot(
            CoI - Q, 0.5 * np.cross(T[4, :, :] - Q, T[2, :, :] - Q)
        ) / np.linalg.norm(
            0.5 * np.cross(T[4, :, :] - Q, T[2, :, :] - Q), axis=1
        )
        hoJ = eps + tdot(
            CoJ - Q, 0.5 * np.cross(T[2, :, :] - Q, T[0, :, :] - Q)
        ) / np.linalg.norm(
            0.5 * np.cross(T[2, :, :] - Q, T[0, :, :] - Q), axis=1
        )
        hoK = eps + tdot(
            CoK - Q, 0.5 * np.cross(T[0, :, :] - Q, T[4, :, :] - Q)
        ) / np.linalg.norm(
            0.5 * np.cross(T[0, :, :] - Q, T[4, :, :] - Q), axis=1
        )

        Kn = np.zeros_like(KK)
        Kt = np.zeros_like(KK)
        Kto = np.zeros_like(KK)
        Kno = np.zeros_like(KK)
        Do1 = np.zeros_like(KK)
        D1 = np.zeros_like(KK)
        D2 = np.zeros_like(KK)
        Do2 = np.zeros_like(KK)
        sigmao = np.zeros_like(KK)
        neubf = np.zeros_like(KK)

        for i in range(6):
            tau1 = np.cross(Nf[i], T[t[i]] - Q)
            tau2 = np.cross(Nf[i], T[u[i]] - Q)
            Kn[i, :] = mdot(Nf[i], Kk, Nf[i]) / (area[i] * area[i])
            Kt[i, :] = mdot(Nf[i], Kk, tau1) / (area[i] * area[i])
            D1[i, :] = tdot(-tau1, Q - C) / (2 * height[i] * area[i])
            D2[i, :] = tdot(-tau2, Q - C) / (2 * height[i] * area[i])

            if i == 0 or i == 1:
                Kto[i, :] = mdot(Nf[i], KkoJ, tau1) / (area[i] * area[i])
                Kno[i, :] = mdot(Nf[i], KkoJ, Nf[i]) / (area[i] * area[i])
                Do1[i, :] = tdot(-tau1, CoJ - Q) / (2 * hoJ * area[i])
                Do2[i, :] = tdot(-tau2, CoJ - Q) / (2 * hoJ * area[i])
                sigmao[i, :] = (area[i] / hoJ) * (
                    mdot(Nf[i], KkoJ, Nf[i]) / (area[i] * area[i])
                )
                neubf[i, :] = (
                    area[i]
                    * valueopJ
                    * (self.node_flag[node] > 200)
                    * (ocJ < 0)
                )
            if i == 2 or i == 3:
                Kto[i, :] = mdot(Nf[i], KkoI, tau1) / (area[i] * area[i])
                Kno[i, :] = mdot(Nf[i], KkoI, Nf[i]) / (area[i] * area[i])
                Do1[i, :] = tdot(-tau1, CoI - Q) / (2 * hoI * area[i])
                Do2[i, :] = tdot(-tau2, CoI - Q) / (2 * hoI * area[i])
                sigmao[i, :] = (area[i] / hoI) * (
                    mdot(Nf[i], KkoI, Nf[i]) / (area[i] * area[i])
                )
                neubf[i, :] = (
                    area[i]
                    * valueopI
                    * (self.node_flag[node] > 200)
                    * (ocI < 0)
                )
            if i == 4 or i == 5:
                Kto[i, :] = mdot(Nf[i], KkoK, tau1) / (area[i] * area[i])
                Kno[i, :] = mdot(Nf[i], KkoK, Nf[i]) / (area[i] * area[i])
                Do1[i, :] = tdot(-tau1, CoK - Q) / (2 * hoK * area[i])
                Do2[i, :] = tdot(-tau2, CoK - Q) / (2 * hoK * area[i])
                sigmao[i, :] = (area[i] / hoK) * (
                    mdot(Nf[i], KkoK, Nf[i]) / (area[i] * area[i])
                )
                neubf[i, :] = (
                    area[i]
                    * valueopK
                    * (self.node_flag[node] > 200)
                    * (ocK < 0)
                )

        sigma = np.array([(area / height) * Kn, sigmao])

        neta = np.array(
            [
                D1 * Kn - Do1 * Kno + 0.5 * Kt - 0.5 * Kto,
                D2 * Kn - Do2 * Kno + 0.5 * Kt - 0.5 * Kto,
            ]
        )

        return neta, sigma, neubf

    def calc_lam(self, node, csi, neta, sigma, neubf, ocI, ocJ, ocK, esurn):

        lam = np.array([0.0] * len(esurn))
        gam = 0
        t = list(range(6))
        u = [1, 2, 3, 4, 5, 0]

        if self.node_flag[node] > 200:
            for i in range(len(esurn)):
                H = np.array([[0.0] * 6 for y in range(6)])
                C = np.array([[0.0] * 4 for y in range(6)])
                H[t, t] = -neta[1, :, i]
                H[t, u] = neta[0, :, i]
                B = np.linalg.inv(H)
                A = csi[:, i]
                C[:, 0] = -sigma[0, :, i]
                C[0:2, 1] = -sigma[1, 0:2, i]
                C[2:4, 2] = -sigma[1, 2:4, i]
                C[4:6, 3] = -sigma[1, 4:6, i]
                L = (A @ B) @ C
                lam[esurn == esurn[i]] += L[0]
                lam[esurn == ocJ[i]] += L[1]
                lam[esurn == ocI[i]] += L[2]
                lam[esurn == ocK[i]] += L[3]
                gam += sum((A @ B) * neubf[:, i]) + sum(neubf[:, i])

        return lam, gam

    # import pdb; pdb.set_trace()
