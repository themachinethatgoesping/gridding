# SPDX-FileCopyrightText: 2022 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

# test basic imports
from themachinethatgoesping.gridding.functions import gridfunctions as grdf

import numpy as np
from pytest import approx


#define class for grouping (test sections)
class Test_echogrids_functions_gridfunctions:
    #define actual tests (must start with "test_"
    #test case 1
    def test_get_minmax_should_return_as_numpy_equivalent(self):
        size = 10

        sx = (np.random.random(size))*100
        sy = (np.random.random(size)-1.0)*100
        sz = (np.random.random(size)-0.5)*100

        for grd_result, np_result in zip(grdf.get_minmax(sx, sy, sz), [np.min(sx), np.max(sx), np.min(sy), np.max(sy), np.min(sz), np.max(sz)]):
            assert grd_result == approx(np_result)
