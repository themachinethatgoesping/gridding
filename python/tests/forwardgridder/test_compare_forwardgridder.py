import pytest
import numpy as np
from themachinethatgoesping.algorithms_cppy import gridding as alg
import themachinethatgoesping.gridding.forwardgridderlegacynew as grd


@pytest.fixture
def sample_data():
    # Generate random test data
    size = 100
    sx = (np.random.random(size)) * 100
    sy = (np.random.random(size) - 1.0) * 100
    sz = (np.random.random(size) - 0.5) * 100
    sv = np.random.random(size) * 10
    return sx, sy, sz, sv


@pytest.fixture
def sample_gridders(sample_data):
    # Create equivalent gridders with the same parameters
    sx, sy, sz, _ = sample_data
    res = 5.0

    # Create from_data
    py_gridder = grd.ForwardGridderLegacyNew.from_data(res, sx, sy, sz)
    cpp_gridder = alg.ForwardGridder3D.from_data(res, sx, sy, sz)

    return py_gridder, cpp_gridder


def test_from_data_equivalence(sample_data):
    # Test that from_data factory creates equivalent gridders
    sx, sy, sz, _ = sample_data
    res = 5.0

    py_gridder = grd.ForwardGridderLegacyNew.from_data(res, sx, sy, sz)
    cpp_gridder = alg.ForwardGridder3D.from_data(res, sx, sy, sz)

    # Compare basic properties
    assert py_gridder.xres == pytest.approx(cpp_gridder.get_xres())
    assert py_gridder.yres == pytest.approx(cpp_gridder.get_yres())
    assert py_gridder.zres == pytest.approx(cpp_gridder.get_zres())
    assert py_gridder.nx == cpp_gridder.get_nx()
    assert py_gridder.ny == cpp_gridder.get_ny()
    assert py_gridder.nz == cpp_gridder.get_nz()
    assert py_gridder.xmin == pytest.approx(cpp_gridder.get_xmin())
    assert py_gridder.xmax == pytest.approx(cpp_gridder.get_xmax())
    assert py_gridder.ymin == pytest.approx(cpp_gridder.get_ymin())
    assert py_gridder.ymax == pytest.approx(cpp_gridder.get_ymax())
    assert py_gridder.zmin == pytest.approx(cpp_gridder.get_zmin())
    assert py_gridder.zmax == pytest.approx(cpp_gridder.get_zmax())


def test_from_res_equivalence():
    # Test that from_res factory creates equivalent gridders
    res = 2.5
    min_x, max_x = 0.0, 100.0
    min_y, max_y = -50.0, 50.0
    min_z, max_z = -25.0, 25.0

    py_gridder = grd.ForwardGridderLegacyNew.from_res(res, min_x, max_x, min_y, max_y, min_z, max_z)
    cpp_gridder = alg.ForwardGridder3D.from_res(res, min_x, max_x, min_y, max_y, min_z, max_z)

    # Compare basic properties
    assert py_gridder.nx == cpp_gridder.get_nx()
    assert py_gridder.ny == cpp_gridder.get_ny()
    assert py_gridder.nz == cpp_gridder.get_nz()
    assert py_gridder.xmin == pytest.approx(cpp_gridder.get_xmin())
    assert py_gridder.xmax == pytest.approx(cpp_gridder.get_xmax())
    assert py_gridder.ymin == pytest.approx(cpp_gridder.get_ymin())
    assert py_gridder.ymax == pytest.approx(cpp_gridder.get_ymax())
    assert py_gridder.zmin == pytest.approx(cpp_gridder.get_zmin())
    assert py_gridder.zmax == pytest.approx(cpp_gridder.get_zmax())


def test_get_minmax_equivalence(sample_data):
    # Test static get_minmax function
    sx, sy, sz, _ = sample_data

    py_result = grd.ForwardGridderLegacyNew.get_minmax(sx, sy, sz)
    cpp_result = alg.ForwardGridder3D.get_minmax(sx, sy, sz)

    # Convert cpp_result tuple to list for easier comparison
    cpp_result_list = list(cpp_result)

    # Compare results
    assert len(py_result) == len(cpp_result_list)
    for py_val, cpp_val in zip(py_result, cpp_result_list):
        assert py_val == pytest.approx(cpp_val)


def test_index_methods_equivalence(sample_gridders):
    # Test index calculation methods
    py_gridder, cpp_gridder = sample_gridders

    # Test values to check
    x_vals = np.array([py_gridder.xmin, py_gridder.xmax, (py_gridder.xmin + py_gridder.xmax) / 2])
    y_vals = np.array([py_gridder.ymin, py_gridder.ymax, (py_gridder.ymin + py_gridder.ymax) / 2])
    z_vals = np.array([py_gridder.zmin, py_gridder.zmax, (py_gridder.zmin + py_gridder.zmax) / 2])

    # Test get_x_index, get_y_index, get_z_index
    for x in x_vals:
        assert py_gridder.get_x_index(x) == cpp_gridder.get_x_index(x)

    for y in y_vals:
        assert py_gridder.get_y_index(y) == cpp_gridder.get_y_index(y)

    for z in z_vals:
        assert py_gridder.get_z_index(z) == cpp_gridder.get_z_index(z)

    # Test get_x_index_fraction, get_y_index_fraction, get_z_index_fraction
    for x in x_vals:
        assert py_gridder.get_x_index_fraction(x) == pytest.approx(cpp_gridder.get_x_index_fraction(x))

    for y in y_vals:
        assert py_gridder.get_y_index_fraction(y) == pytest.approx(cpp_gridder.get_y_index_fraction(y))

    for z in z_vals:
        assert py_gridder.get_z_index_fraction(z) == pytest.approx(cpp_gridder.get_z_index_fraction(z))


def test_value_methods_equivalence(sample_gridders):
    # Test value calculation methods
    py_gridder, cpp_gridder = sample_gridders

    # Test indices to check
    x_indices = [0, py_gridder.nx // 2, py_gridder.nx - 1]
    y_indices = [0, py_gridder.ny // 2, py_gridder.ny - 1]
    z_indices = [0, py_gridder.nz // 2, py_gridder.nz - 1]

    # Test get_x_value, get_y_value, get_z_value
    for i in x_indices:
        assert py_gridder.get_x_value(i) == pytest.approx(cpp_gridder.get_x_value(i))

    for i in y_indices:
        assert py_gridder.get_y_value(i) == pytest.approx(cpp_gridder.get_y_value(i))

    for i in z_indices:
        assert py_gridder.get_z_value(i) == pytest.approx(cpp_gridder.get_z_value(i))

    # Test values to check
    x_vals = np.array([py_gridder.xmin, py_gridder.xmax, (py_gridder.xmin + py_gridder.xmax) / 2])
    y_vals = np.array([py_gridder.ymin, py_gridder.ymax, (py_gridder.ymin + py_gridder.ymax) / 2])
    z_vals = np.array([py_gridder.zmin, py_gridder.zmax, (py_gridder.zmin + py_gridder.zmax) / 2])

    # Test get_x_grd_value, get_y_grd_value, get_z_grd_value
    for x in x_vals:
        assert py_gridder.get_x_grd_value(x) == pytest.approx(cpp_gridder.get_x_grd_value(x))

    for y in y_vals:
        assert py_gridder.get_y_grd_value(y) == pytest.approx(cpp_gridder.get_y_grd_value(y))

    for z in z_vals:
        assert py_gridder.get_z_grd_value(z) == pytest.approx(cpp_gridder.get_z_grd_value(z))


def test_extent_methods_equivalence(sample_gridders):
    # Test extent calculation methods
    py_gridder, cpp_gridder = sample_gridders

    # Test get_extent_x, get_extent_y, get_extent_z
    py_extent_x = py_gridder.get_extent_x()
    cpp_extent_x = cpp_gridder.get_extent_x()
    assert len(py_extent_x) == len(cpp_extent_x)
    for py_val, cpp_val in zip(py_extent_x, cpp_extent_x):
        assert py_val == pytest.approx(cpp_val)

    py_extent_y = py_gridder.get_extent_y()
    cpp_extent_y = cpp_gridder.get_extent_y()
    assert len(py_extent_y) == len(cpp_extent_y)
    for py_val, cpp_val in zip(py_extent_y, cpp_extent_y):
        assert py_val == pytest.approx(cpp_val)

    py_extent_z = py_gridder.get_extent_z()
    cpp_extent_z = cpp_gridder.get_extent_z()
    assert len(py_extent_z) == len(cpp_extent_z)
    for py_val, cpp_val in zip(py_extent_z, cpp_extent_z):
        assert py_val == pytest.approx(cpp_val)

    # Test get_extent with different axis combinations
    for axis in ["x", "y", "z", "xy", "xz", "yz", "xyz"]:
        py_extent = py_gridder.get_extent(axis)
        cpp_extent = cpp_gridder.get_extent(axis)
        assert len(py_extent) == len(cpp_extent)
        for py_val, cpp_val in zip(py_extent, cpp_extent):
            assert py_val == pytest.approx(cpp_val)


def test_coordinates_methods_equivalence(sample_gridders):
    # Test coordinate calculation methods
    py_gridder, cpp_gridder = sample_gridders

    # Test get_x_coordinates, get_y_coordinates, get_z_coordinates
    py_x_coords = py_gridder.get_x_coordinates()
    cpp_x_coords = cpp_gridder.get_x_coordinates()
    assert len(py_x_coords) == len(cpp_x_coords)
    for py_val, cpp_val in zip(py_x_coords, cpp_x_coords):
        assert py_val == pytest.approx(cpp_val)

    py_y_coords = py_gridder.get_y_coordinates()
    cpp_y_coords = cpp_gridder.get_y_coordinates()
    assert len(py_y_coords) == len(cpp_y_coords)
    for py_val, cpp_val in zip(py_y_coords, cpp_y_coords):
        assert py_val == pytest.approx(cpp_val)

    py_z_coords = py_gridder.get_z_coordinates()
    cpp_z_coords = cpp_gridder.get_z_coordinates()
    assert len(py_z_coords) == len(cpp_z_coords)
    for py_val, cpp_val in zip(py_z_coords, cpp_z_coords):
        assert py_val == pytest.approx(cpp_val)


def test_block_mean_interpolation_equivalence(sample_data, sample_gridders):
    # Test block mean interpolation
    sx, sy, sz, sv = sample_data
    py_gridder, cpp_gridder = sample_gridders

    # Create empty grids
    py_values, py_weights = py_gridder.get_empty_grd_images()
    cpp_values, cpp_weights = cpp_gridder.get_empty_grd_images()

    # Run interpolation
    py_gridder.interpolate_block_mean(sx, sy, sz, sv, py_values, py_weights)
    cpp_gridder.interpolate_block_mean_inplace(sx, sy, sz, sv, cpp_values, cpp_weights)

    # Compare results
    # Only compare non-zero elements to avoid precision issues with zeros
    mask = cpp_weights > 0

    assert np.all(np.isclose(py_values[mask], cpp_values[mask], rtol=1e-5, atol=1e-8))
    assert np.all(np.isclose(py_weights[mask], cpp_weights[mask], rtol=1e-5, atol=1e-8))


def test_weighted_mean_interpolation_equivalence(sample_data, sample_gridders):
    # Test weighted mean interpolation
    sx, sy, sz, sv = sample_data
    py_gridder, cpp_gridder = sample_gridders

    # Create empty grids
    py_values, py_weights = py_gridder.get_empty_grd_images()
    cpp_values, cpp_weights = cpp_gridder.get_empty_grd_images()

    # Run interpolation
    py_gridder.interpolate_weighted_mean(sx, sy, sz, sv, py_values, py_weights)
    cpp_gridder.interpolate_weighted_mean_inplace(sx, sy, sz, sv, cpp_values, cpp_weights)

    # Compare results
    # Only compare non-zero elements to avoid precision issues with zeros
    mask = cpp_weights > 0

    assert np.all(np.isclose(py_values[mask], cpp_values[mask], rtol=1e-5, atol=1e-8))
    assert np.all(np.isclose(py_weights[mask], cpp_weights[mask], rtol=1e-5, atol=1e-8))
