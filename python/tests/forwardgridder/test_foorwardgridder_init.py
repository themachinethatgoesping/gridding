# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: MPL-2.0

# test basic imports
from themachinethatgoesping.gridding.forwardgridderlegacynew import ForwardGridderLegacyNew

from pytest import approx


# test function to test the grid parameter initialization for the ForwardGridderLegacyNew class
class Test_ForwardGridder_init:
    # define actual tests (must start with "test_"
    def test_ForwardGridder_init_should_reproduce_expected_grid_parameters(self):
        res = (1, 1, 1)
        xmin = -2.4
        ymin = -2.5
        zmin = -2.6
        xmax = 3.5
        ymax = 3.4
        zmax = 3.6

        gridder = ForwardGridderLegacyNew(*res, xmin, xmax, ymin, ymax, zmin, zmax)

        assert gridder.xmin == -2
        assert gridder.ymin == -3
        assert gridder.zmin == -3

        assert gridder.xmax == 4
        assert gridder.ymax == 3
        assert gridder.zmax == 4

        assert gridder.nx == 7
        assert gridder.ny == 7
        assert gridder.nz == 8

        assert gridder.border_xmin == approx(-2.5)
        assert gridder.border_xmax == approx(4.5)
        assert gridder.border_ymin == approx(-3.5)
        assert gridder.border_ymax == approx(3.5)
        assert gridder.border_zmin == approx(-3.5)
        assert gridder.border_zmax == approx(4.5)

    def test_ForwardGridder_init_should_reproduce_expected_grid_parameters_for_different_base(
        self,
    ):
        res = (1, 1, 1)
        xmin = -2.4
        ymin = -2.5
        zmin = -2.6
        xmax = 3.5
        ymax = 3.4
        zmax = 3.6
        xbase = 0.5
        ybase = -0.5
        zbase = 1 / 3

        gridder = ForwardGridderLegacyNew(
            *res, xmin, xmax, ymin, ymax, zmin, zmax, xbase, ybase, zbase
        )

        assert gridder.xmin == -2.5
        assert gridder.ymin == -2.5
        assert gridder.zmin == approx(-2 - 2 / 3)

        assert gridder.xmax == 3.5
        assert gridder.ymax == 3.5
        assert gridder.zmax == approx(3 + 1 / 3)

        assert gridder.nx == 7
        assert gridder.ny == 7
        assert gridder.nz == 7

        assert gridder.border_xmin == approx(-3)
        assert gridder.border_xmax == approx(4)
        assert gridder.border_ymin == approx(-3)
        assert gridder.border_ymax == approx(4)
        assert gridder.border_zmin == approx(-2 - 2 / 3 - 0.5)
        assert gridder.border_zmax == approx(3 + 1 / 3 + 0.5)

    def test_ForwardGridder_init_should_reproduce_expected_grid_parameters_for_different_res(
        self,
    ):
        res = (0.5, 1 / 3, 2)
        xmin = -2.4
        ymin = -2.5
        zmin = -2.6
        xmax = 3.5
        ymax = 3.4
        zmax = 3.6

        gridder = ForwardGridderLegacyNew(*res, xmin, xmax, ymin, ymax, zmin, zmax)

        assert gridder.xmin == approx(-2.5)
        assert gridder.ymin == approx(-2 - 2 / 3)
        assert gridder.zmin == approx(-2)

        assert gridder.xmax == approx(3.5)
        assert gridder.ymax == approx(3 + 1 / 3)
        assert gridder.zmax == approx(4)

        assert gridder.nx == 13
        assert gridder.ny == 19
        assert gridder.nz == 4

        assert gridder.border_xmin == approx(-2.75)
        assert gridder.border_xmax == approx(3.75)
        assert gridder.border_ymin == approx(-2 - 2 / 3 - 1 / 6)
        assert gridder.border_ymax == approx(3 + 1 / 3 + 1 / 6)
        assert gridder.border_zmin == approx(-3)
        assert gridder.border_zmax == approx(5)
