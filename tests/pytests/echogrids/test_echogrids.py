# SPDX-FileCopyrightText: 2022 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

# test basic imports
from themachinethatgoesping import echogrids
from themachinethatgoesping.echogrids import EchoGrid
from themachinethatgoesping.echogrids import functions
from themachinethatgoesping.echogrids.functions import gridfunctions
from themachinethatgoesping.echogrids.functions.gridfunctions import GRIDDER

import time
import pytest


#define class for grouping (test sections)
class Test_echogrids_template:
    #define actual tests (must start with "test_"
    #test case 1
    def test_tests_should_run(self):
        assert 1 == 1



