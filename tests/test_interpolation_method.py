import numpy as np
import pytest

from mpfad.interpolation.InterpolationMethod import InterpolationMethodBase
from mpfad.interpolation.IDW import IDW
from mpfad.interpolation.LSW import LSW


@pytest.fixture
def get_mocked_method(mocker):
    mocker.patch('mpfad.interpolation.InterpolationMethod.InterpolationMethodBase._set_nodes')
    mocker.patch('mpfad.interpolation.InterpolationMethod.InterpolationMethodBase._set_faces')

    def get_method(method_class):
        return method_class(mocker.Mock())

    return get_method


@pytest.fixture
def base_method(get_mocked_method):
    return get_mocked_method(InterpolationMethodBase)


@pytest.fixture
def idw_method(get_mocked_method):
    return get_mocked_method(IDW)


@pytest.fixture
def lsw_method(get_mocked_method):
    return get_mocked_method(LSW)


class TestInterpolationMethodBase:

    def test_interpolate_raises_NotImplementedError(self, base_method):
        with pytest.raises(NotImplementedError):
            base_method.interpolate(None)


class TestIDW:

    def test_calc_weight(self, idw_method):
        node_coords = np.array([0., 0., 0.])
        vol_centroid = np.array([4., 0., 0.])

        idw_method.mesh_data.get_centroid.return_value = vol_centroid
        weight = idw_method.calc_weight(node_coords, None)

        assert weight == pytest.approx(0.25)

    def test_interpolate(self, idw_method):
        node_coords = np.array([0., 0., 0.])
        vols_centroids = [np.array([4., 0., 0.]), np.array([8., 0., 0.]),
                          np.array([0., 4., 0.])]

        idw_method.mtu.get_bridge_adjacencies.return_value = range(3)
        idw_method.mesh_data.get_centroid.side_effect = vols_centroids
        idw_method.mb.get_coords.return_value = node_coords

        weight = idw_method.interpolate(None)

        assert weight[0] == pytest.approx(0.4)
        assert weight[1] == pytest.approx(0.2)
        assert weight[2] == pytest.approx(0.4)


class TestLSW:

    def test_calc_G(self, lsw_method):
        node_coords = np.array([0., 0., 0.])
        vol_centroid = np.array([4., 0., 0.])

        lsw_method.mesh_data.get_centroid.return_value = vol_centroid
        lsw_method.mesh_data.get_coords.return_value = node_coords

        G = lsw_method.calc_G(59., -13., 9, 11., 1., 3.,)

        assert G == pytest.approx(256.)

    def test_calc_psi(self, lsw_method):
        node_coords = np.array([0., 0., 0.])
        vol_centroid = np.array([1., 0., 0.])

        lsw_method.mesh_data.get_centroid.return_value = vol_centroid
        lsw_method.mesh_data.get_coords.return_value = node_coords
        R_x, R_y, R_z = 1.5, 1.58333333333333, 4.
        I_xx, I_yy, I_zz = 1.125, 0.923611111111111,	6
        I_xy, I_xz, I_yz = 0.958333333333333,	1.75, 1.91666666666667
        G = 0.19140625

        psi_x = lsw_method.calc_psi(R_x, R_y, R_z,
                                    I_xy, I_xz, I_yy,
                                    I_yz, I_zz, G)
        psi_y = lsw_method.calc_psi(R_y, R_x, R_z,
                                    I_xy, I_yz, I_xx,
                                    I_xz, I_zz, G)
        psi_z = lsw_method.calc_psi(R_z, R_x, R_y,
                                    I_xz, I_yz, I_xx,
                                    I_xy, I_yy, G)

        assert psi_x == pytest.approx(0.571428571428568)
        assert psi_y == pytest.approx(-1.71428571428571000000)
        assert psi_z == pytest.approx(-0.285714285714283)

    def test_interpolate(self, lsw_method):
        node_coords = np.array([0., 0., 0.])
        vols_centroids = [np.array([.1, .5, 3.]), np.array([1., 1., 1.]),
                          np.array([0.25, .25, 2.])]

        lsw_method.mtu.get_bridge_adjacencies.return_value = range(3)
        lsw_method.mesh_data.get_centroid.side_effect = vols_centroids
        lsw_method.mb.get_coords.return_value = node_coords

        weight = lsw_method.interpolate(None)
        # print(weight)

        assert weight[0] == pytest.approx(1/3.)
        assert weight[1] == pytest.approx(4/9.)
        assert weight[2] == pytest.approx(1/5.)
