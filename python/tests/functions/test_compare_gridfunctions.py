import pytest
import numpy as np
import themachinethatgoesping.algorithms.gridding.functions as algf
import themachinethatgoesping.gridding.functions.gridfunctions as grdgf

def test_minmax_equivalence():
    sx = np.array([1.0, 2.5, 3.5])
    sy = np.array([10.2, 5.3, 9.1])
    sz = np.array([-1.0, -5.0, 0.0])

    result_gridding = grdgf.get_minmax(sx, sy, sz)
    result_algorithms = algf.get_minmax(sx, sy, sz)
    assert result_gridding == result_algorithms, "Results differ between gridding and algorithms get_minmax"

def test_get_index_equivalence():
    xvals = np.array([-5.0, -1.2, 0.0, 1.7, 3.9, 10.5])
    xmin = -5.0
    xres = 1.0
    for x in xvals:
        result_gridding = grdgf.get_index(x, xmin, xres)
        result_algorithms = algf.get_index(x, xmin, xres)
        assert result_gridding == result_algorithms, f"get_index differs for value {x}"

def test_get_index_fraction_equivalence():
    xvals = np.array([-5.0, -1.2, 0.0, 1.7, 3.9, 10.5])
    xmin = -5.0
    xres = 1.0
    for x in xvals:
        result_gridding = grdgf.get_index_fraction(x, xmin, xres)
        result_algorithms = algf.get_index_fraction(x, xmin, xres)
        assert result_gridding == pytest.approx(result_algorithms, abs=1e-8, rel=1e-5), \
            f"get_index_fraction differs for value {x}: {result_gridding} != {result_algorithms}"

def test_get_value_equivalence():
    indices = np.array([0, 1, 5, 10, 15, 20])
    min_val = -5.0
    res = 0.5
    for idx in indices:
        result_gridding = grdgf.get_value(idx, min_val, res)
        result_algorithms = algf.get_value(idx, min_val, res)
        assert result_gridding == pytest.approx(result_algorithms, abs=1e-8, rel=1e-5), \
            f"get_value differs for index {idx}: {result_gridding} != {result_algorithms}"

def test_get_grd_value_equivalence():
    values = np.array([-5.0, -4.7, -3.2, 0.0, 2.8, 5.5])
    min_val = -5.0
    res = 0.5
    for val in values:
        result_gridding = grdgf.get_grd_value(val, min_val, res)
        result_algorithms = algf.get_grd_value(val, min_val, res)
        assert result_gridding == pytest.approx(result_algorithms, abs=1e-8, rel=1e-5), \
            f"get_grd_value differs for value {val}: {result_gridding} != {result_algorithms}"

def test_get_index_weights_equivalence():
    # Test a few different combinations of fractional coordinates
    test_cases = [
        (1.5, 2.7, 3.2),
        (0.3, 0.8, 1.9),
        (5.0, 6.0, 7.0),
        (2.4, 1.6, 3.8)
    ]
    
    for frac_x, frac_y, frac_z in test_cases:
        result_gridding = grdgf.get_index_weights(frac_x, frac_y, frac_z)
        result_algorithms = algf.get_index_weights(frac_x, frac_y, frac_z)
        
        # Compare each component separately for clearer error messages
        assert np.array_equal(result_gridding[0], result_algorithms[0]), \
            f"X indices differ for fractional coords ({frac_x}, {frac_y}, {frac_z})"
        assert np.array_equal(result_gridding[1], result_algorithms[1]), \
            f"Y indices differ for fractional coords ({frac_x}, {frac_y}, {frac_z})"
        assert np.array_equal(result_gridding[2], result_algorithms[2]), \
            f"Z indices differ for fractional coords ({frac_x}, {frac_y}, {frac_z})"
        for w1, w2 in zip(result_gridding[3], result_algorithms[3]):
            assert w1 == pytest.approx(w2, abs=1e-8, rel=1e-5), \
                f"Weights differ for fractional coords ({frac_x}, {frac_y}, {frac_z})"
                

def test_grd_weighted_mean_should_equal_cpp_implementation():
    # Generate random test data
    size = 100
    sx = (np.random.random(size)) * 100
    sy = (np.random.random(size) - 1.0) * 100
    sz = (np.random.random(size) - 0.5) * 100
    sv = np.random.random(size) * 10
    
    # Grid parameters
    xmin, xres, nx = 0.0, 5.0, 30
    ymin, yres, ny = -100.0, 5.0, 30
    zmin, zres, nz = -50.0, 5.0, 20
    
    # Create arrays for results
    py_values = np.zeros((nx, ny, nz), dtype=np.float64)
    py_weights = np.zeros((nx, ny, nz), dtype=np.float64)
    cpp_values = np.zeros((nx, ny, nz), dtype=np.float64)
    cpp_weights = np.zeros((nx, ny, nz), dtype=np.float64)
    
    # Apply both implementations
    grdgf.grd_weighted_mean(
        sx, sy, sz, sv,
        xmin, xres, nx,
        ymin, yres, ny,
        zmin, zres, nz,
        py_values, py_weights
    )
    
    algf.grd_weighted_mean(
        sx, sy, sz, sv,
        xmin, xres, nx,
        ymin, yres, ny,
        zmin, zres, nz,
        cpp_values, cpp_weights
    )
    
    # Check results are equivalent
    # Only compare non-zero elements to avoid precision issues with zeros
    mask = (cpp_weights > 0)
    
    assert np.all(np.isclose(py_values[mask], cpp_values[mask], rtol=1e-5, atol=1e-8))
    assert np.all(np.isclose(py_weights[mask], cpp_weights[mask], rtol=1e-5, atol=1e-8))

def test_grd_block_mean_should_equal_cpp_implementation():
    # Generate random test data
    size = 100
    sx = (np.random.random(size)) * 100
    sy = (np.random.random(size) - 1.0) * 100
    sz = (np.random.random(size) - 0.5) * 100
    sv = np.random.random(size) * 10
    
    # Grid parameters
    xmin, xres, nx = 0.0, 5.0, 30
    ymin, yres, ny = -100.0, 5.0, 30
    zmin, zres, nz = -50.0, 5.0, 20
    
    # Create arrays for results
    py_values = np.zeros((nx, ny, nz), dtype=np.float64)
    py_weights = np.zeros((nx, ny, nz), dtype=np.float64)
    cpp_values = np.zeros((nx, ny, nz), dtype=np.float64)
    cpp_weights = np.zeros((nx, ny, nz), dtype=np.float64)
    
    # Apply both implementations
    grdgf.grd_block_mean(
        sx, sy, sz, sv,
        xmin, xres, nx,
        ymin, yres, ny,
        zmin, zres, nz,
        py_values, py_weights
    )
    
    algf.grd_block_mean(
        sx, sy, sz, sv,
        xmin, xres, nx,
        ymin, yres, ny,
        zmin, zres, nz,
        cpp_values, cpp_weights
    )
    
    # Check results are equivalent
    # Only compare non-zero elements to avoid precision issues with zeros
    mask = (cpp_weights > 0)
    
    assert np.all(np.isclose(py_values[mask], cpp_values[mask], rtol=1e-5, atol=1e-8))
    assert np.all(np.isclose(py_weights[mask], cpp_weights[mask], rtol=1e-5, atol=1e-8))

