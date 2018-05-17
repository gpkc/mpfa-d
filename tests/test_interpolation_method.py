import numpy as np
import pytest

from mpfad.interpolation.InterpolationMethod import InterpolationMethodBase
from mpfad.interpolation.IDW import IDW


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
