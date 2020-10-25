# for i in self.T_expanded.shape
# inner_ids = []
# boundary_ids = []
# [
#     inner_ids.append(self.mb.tag_get_data(self.global_id_tag, volume))
#     if len(self.mtu.get_bridge_adjacencies(volume, 2, 3)) == 4
#     else boundary_ids.append(
#         self.mb.tag_get_data(self.global_id_tag, volume)
#     ) for volume in self.volumes
# ]
# inner_adjacencies_ids = []
# boundary_adjacencies_ids = []
# [
#     inner_adjacencies_ids.append(
#         self.mb.tag_get_data(
#             self.global_id_tag,
#             self.mtu.get_bridge_adjacencies(volume, 2, 3)
#         )
#     )
#     if len(self.mtu.get_bridge_adjacencies(volume, 2, 3)) == 4
#     else boundary_adjacencies_ids.append(
#         self.mb.tag_get_data(
#             self.global_id_tag,
#             self.mtu.get_bridge_adjacencies(volume, 2, 3)
#         )
#     ) for volume in self.volumes
# ]
# for row, columns in zip(inner_ids, inner_adjacencies_ids):
#     i = row[0][0]
#     for column in columns:
#         j = column[0]
#         if i != j:
#             self.T_plus[i, j] = max(0, self.T[i][j])
#         elif i == j:
#             self.T_plus[i, j] = -np.sum(
#                 [t for t in self.T[i]]
#             )
#     self.T_minus[i, :] = self.T[i] - self.T_plus[i]
#     self._T[i, :] = self.T[i]
# for row, columns in zip(boundary_ids, boundary_adjacencies_ids):
#     i = row[0][0]
#     for column in columns:
#         j = column[0]
#         if i != j:
#             self.T_plus[i, j] = max(0, self.T[i][j])
#         elif i == j:
#             self.T_plus[i, j] = -np.sum(
#                 [t for t in self.T[i]]
#             )
#     self.T_minus[i, :] = self.T[i] - self.T_plus[i]
#     self._T[i, :] = self.T[i]
# from openpyxl import Workbook
# from openpyxl.styles import Color, Fill
# wb = Workbook()
# ws = wb.active
# ws.title = "RESULTS"
# rows, columns, values = find(self.T_minus)
# for i, j, v in zip(rows, columns, values):
#     ws.cell(row=i + 1, column=j + 1, value=v)
# ids = []
# boundary_ids = []
# [
#     ids.append(self.mb.tag_get_data(self.global_id_tag, volume))
#     for volume in self.volumes
# ]
# adjacencies_ids = []
# [
#     adjacencies_ids.append(
#         self.mb.tag_get_data(
#             self.global_id_tag,
#             self.mtu.get_bridge_adjacencies(volume, 2, 3)
#         )
#     )
#     for volume in self.volumes
# ]
# for row, columns in zip(ids, adjacencies_ids):
#     i = row[0][0]
#     for column in columns:
#         j = column[0]
#         if i != j:
#             self.T_plus[i, j] = max(0, self.T[i][j])
#         elif i == j:
#             self.T_plus[i, j] = -np.sum(
#                 [t for t in self.T[i]]
#             )
#     self.T_minus[i, :] = self.T[i] - self.T_plus[i]
#     self._T[i, :] =  self.T[i]
# from openpyxl import Workbook
# wb = Workbook()
# ws = wb.active
# ws.title = "RESULTS"
# rows, columns, values = find(self.T_minus)
# for i, j, v in zip(rows, columns, values):
#     ws.cell(row=i + 1, column=j + 1, value=v)
# wb.save('8x8x8_T_miuns.xlsx')
# self.boundary_ids = [bid[0][0] for bid in boundary_ids]
# self.q = self.Q.tocsc()
# self.x = spsolve(self.T_minus, self.q)
# f = (self.T_plus) * self.x
# q = self.q.toarray()
# residual = q[:, 0] + f - self.T_minus * self.x
# antidiffusive_flux = self.flux_limiter(antidiffusive_flux)
# for volume, vertices in antidiffusive_flux.items():
#     for vertice, vertice_data in vertices[0].items():
#         cumm = 0
#         g, source = vertice_data
#         cumm += g * source
#     q[volume] = - cumm
# residual += 1
# while max(abs(residual)) > 1E-3:
#     self.dx = spsolve(self.T_minus, residual)
#     self.x += self.dx
# alfa = self.slip()
# for row, columns in zip(ids, adjacencies_ids):
#         i = row[0][0]
#         for column in columns:
#             j = column[0]
#             if i != j:
#                 self.T_plus[i, j] = alfa[i, j] * self.T_plus[i, j]
# f = self.T_plus * self.x
# rows, columns, values = find(alfa)
# antidiffusive_flux = self.flux_limiter(antidiffusive_flux)
# for volume, vertices in antidiffusive_flux.items():
#     for vertice, vertice_data in vertices[0].items():
#         cumm = 0
#         g, source = vertice_data
#         cumm += g * source
#     q[volume] = -cumm
# residual = q[:, 0] + f - self.T_minus * self.x
# print(f"max res: {max(abs(residual))}")
# while max(abs(residual)) > 1E-3:
#     print('max residual :', max(abs(residual)))
#     alfa = self.slip()
#     for row, columns in zip(inner_ids, inner_adjacencies_ids):
#         i = row[0][0]
#         for column in columns:
#             j = column[0]
#             if i != j:
#                 self.T_plus[i, j] = alfa[i, j] * self.T_plus[i, j]
#     f = (-self.T_minus + self._T_plus) * self.x
#     residual = q[:, 0] + f
#     self.dx = spsolve(self.T_minus, residual)
#     self.x += self.dx
#     rows, columns, values = find(alfa)
#     vals = [value for value in values if value != 1.0]
#     print(vals)
