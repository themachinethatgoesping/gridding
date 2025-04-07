# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

# test basic imports
from themachinethatgoesping.gridding.forwardgridder import ForwardGridder

import numpy as np
from pytest import approx


# test functions to test the grid indexing functions for the ForwardGridder class
class Test_ForwardGridder_interpolation:
    # TODO: need some real tests for the interpolation ...
    def test_ForwardGridder_should_not_change_the_value_sum(self):
        res = 0.5
        sx = np.array([-1, -1.3, 0, 0, 3])
        sy = np.array([-1, -1.3, 2, 2, 3])
        sz = np.array([-1, -1.3, 0, 0, 3])
        sv = np.array([1, 2, 3, 4, 5]).astype(sx.dtype)

        gridder = ForwardGridder.from_data(res, sx, sy, sz)

        # block mean
        ival, iweight = gridder.interpolate_block_mean(sx, sy, sz, sv)
        assert np.nansum(ival) == approx(np.nansum(sv))
        assert np.nansum(iweight) == approx(len(sv))

        # weight6ed mean
        ival, iweight = gridder.interpolate_weighted_mean(sx, sy, sz, sv)
        assert np.nansum(ival) == approx(np.nansum(sv))
        assert np.nansum(iweight) == approx(len(sv))
