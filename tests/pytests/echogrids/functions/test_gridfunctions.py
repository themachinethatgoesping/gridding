# SPDX-FileCopyrightText: 2022 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

# test basic imports
from themachinethatgoesping.echogrids.functions import gridfunctions

import numpy as np
import time
import pytest


#define class for grouping (test sections)
class Test_echogrids_functions_gridfunctions:
    #define actual tests (must start with "test_"
    #test case 1
    def test_get_minmax_should_return_as_numpy_equivalent(self):
        size = 100

        sx = (np.random.random(100))*100
        sy = (np.random.random(100)-1.0)*100
        sz = (np.random.random(100)-0.5)*100

        for grd_result, np_result in zip(gridfunctions.get_minmax(sx, sy, sz), [np.min(sx), np.max(sx), np.min(sy), np.max(sy), np.min(sz), np.max(sz)]):
            assert grd_result == pytest.approx(np_result)
