# SPDX-FileCopyrightText: 2022 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

# test basic imports
from themachinethatgoesping.gridding.forwardgridder import ForwardGridder

import numpy as np
import time
from pytest import approx


#test functions to test the grid indexing functions for the ForwardGridder class
class Test_ForwardGridder_indexfunctions:
    def test_ForwardGridder_index_functions_should_reproduce_expected_gridparameters(self):
      res = (1, 0.5, 1/3)
      xmin = -2.4
      ymin = -2.5
      zmin = -2.6
      xmax = 3.5
      ymax = 3.4
      zmax = 3.6

      gridder = ForwardGridder(*res, xmin, xmax, ymin, ymax, zmin, zmax)

      #indices
      assert gridder.get_x_index(-2.4) == 0
      assert gridder.get_x_index(-3.4) == -1 #this is out of bounds actually ..
      assert gridder.get_x_index(3) == 5

      assert gridder.get_y_index(-2.4) == 0
      assert gridder.get_y_index(-3.4) == -2 #this is out of bounds actually ..
      assert gridder.get_y_index(3) == 11

      assert gridder.get_z_index(-2.4) == 1
      assert gridder.get_z_index(-3.4) == -2
      assert gridder.get_z_index(3) == 17
    
      #get (block) gridded index
      assert gridder.get_x_grd_value(-2.4) == -2
      assert gridder.get_x_grd_value(-3.4) == -3 #this is out of bounds actually ..
      assert gridder.get_x_grd_value(3) == 3

      assert gridder.get_y_grd_value(-2.4) == -2.5
      assert gridder.get_y_grd_value(-3.4) == -3.5 #this is out of bounds actually ..
      assert gridder.get_y_grd_value(3) == 3

      assert gridder.get_z_grd_value(-2.4) == approx(-2 - 1/3)
      assert gridder.get_z_grd_value(-3.4) == approx(-3 - 1/3)
      assert gridder.get_z_grd_value(3) == approx(3)