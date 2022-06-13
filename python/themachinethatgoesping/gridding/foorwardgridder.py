# SPDX-FileCopyrightText: 2022 Peter Urban, Ghent University
# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

"""
Simple gridder class that can create quantitative 3D images from x,z,y,val from some custom data.
"""

import math
import numpy as np

from .functions import gridfunctions as grdf


class FoorwardGridder(object):
    """ Simple class to generate 3D grids (images) and interpolate xyz data ontothem using simple foorwardmapping algorithms.
    (e.g. block mean, weighted mean interpolation)
    """

    # -- factory methods --
    @classmethod
    def from_res(cls,
                 res: float,
                 min_x: float, max_x: float,
                 min_y: float, max_y: float,
                 min_z: float, max_z: float):
        """Create gridder setting xres,yres and zres to "res"

        Parameters
        ----------
        res : float
            x,y and z resolution of the grid
        min_x : float
            smallest x value that must be contained within the grid
        max_x : float
            largest x value that must be contained within the grid
        min_y : float
            smallest y value that must be contained within the grid
        max_y : float
            smallest y value that must be contained within the grid
        min_z : float
            smallest z value that must be contained within the grid
        max_z : float
            smallest z value that must be contained within the grid

        Returns
        -------
        FoorwardGridder
            FoorwardGridder object
        """
        return cls(res,res,res,min_x,max_x,min_y,max_y,min_z,max_z)


    def __init__(self,
                 xres: float,
                 yres: float,
                 zres: float,
                 min_x: float, max_x: float,
                 min_y: float, max_y: float,
                 min_z: float, max_z: float,
                 xbase: float = None,
                 ybase: float = None,
                 zbase: float = None):

        # resolution in meter
        self.xres = xres
        self.yres = yres
        self.zres = zres

        if xbase is None:
            self.xbase = 0
        else:
            self.xbase = xbase
        if ybase is None:
            self.ybase = 0
        else:
            self.ybase = ybase
        if zbase is None:
            self.zbase = 0
        else:
            self.zbase = zbase

        nx = math.floor((min_x - self.xbase) / xres)
        ny = math.floor((min_y - self.ybase) / yres)
        nz = math.floor((min_z - self.zbase) / zres)

        self.xmin = nx * xres + self.xbase
        self.ymin = ny * yres + self.ybase
        self.zmin = nz * zres + self.zbase

        nx = math.ceil((max_x - self.xmin) / xres)
        ny = math.ceil((max_y - self.ymin) / yres)
        nz = math.ceil((max_z - self.zmin) / zres)

        self.xmax = nx * xres + self.xmin
        self.ymax = ny * yres + self.ymin
        self.zmax = nz * zres + self.zmin

        # with round, the rounding error will be eliminated which cause res=0.3 to crash
        # num of elements x
        self.nx = math.floor(
            round(((self.xmax - self.xmin) / self.xres), 8)) + 1
        # num of elements y
        self.ny = math.floor(
            round(((self.ymax - self.ymin) / self.yres), 8)) + 1
        # num of elements z
        self.nz = math.floor(
            round(((self.zmax - self.zmin) / self.zres), 8)) + 1
        # self.nx=math.floor((self.xmax-self.xmin)/self.res)+1 #num of elements y
        # self.ny=math.floor((self.ymax-self.ymin)/self.res)+1 #num of elements x

        # borders
        self.border_xmin = self.xmin - xres / 2.0
        self.border_xmax = self.xmax + xres / 2.0
        self.border_ymin = self.ymin - yres / 2.0
        self.border_ymax = self.ymax + yres / 2.0
        self.border_zmin = self.zmin - zres / 2.0
        self.border_zmax = self.zmax + zres / 2.0

    def get_x_index(self, val):
        return get_index(val, self.xmin, self.xres)

    def get_y_index(self, val):
        return get_index(val, self.ymin, self.yres)

    def get_z_index(self, val):
        return get_index(val, self.zmin, self.zres)

    def get_x_index_fraction(self, val):
        return get_index_fraction(val, self.xmin, self.xres)

    def get_y_index_fraction(self, val):
        return get_index_fraction(val, self.ymin, self.yres)

    def get_z_index_fraction(self, val):
        return get_index_fraction(val, self.zmin, self.zres)

    def get_x_value(self, index):
        return get_value(index, self.xmin, self.xres)

    def get_y_value(self, index):
        return get_value(index, self.ymin, self.yres)

    def get_z_value(self, index):
        return get_value(index, self.zmin, self.zres)

    def get_x_grd_value(self, value):
        return self.get_x_value(self.get_x_index(value))

    def get_y_grd_value(self, value):
        return self.get_y_value(self.get_y_index(value))

    def get_z_grd_value(self, value):
        return self.get_z_value(self.get_z_index(value))

    def get_extent_x(self):
        return [self.border_xmin, self.border_xmax]

    def get_extent_y(self):
        return [self.border_ymin, self.border_ymax]

    def get_extent_z(self):
        return [self.border_zmin, self.border_zmax]

    def get_min_and_offset(self):
        return self.xmin, self.xres, self.nx, self.ymin, self.yres, self.ny, self.zmin, self.zres, self.nz

    def get_x_coordinates(self):

        coordinates = []
        for i in range(self.nx):
            coordinates.append(self.get_x_value(i))

        return coordinates

    def get_y_coordinates(self):

        coordinates = []
        for i in range(self.ny):
            coordinates.append(self.get_y_value(i))

        return coordinates

    def get_z_coordinates(self):

        coordinates = []
        for i in range(self.nz):
            coordinates.append(self.get_z_value(i))

        return coordinates

    def get_sampled_image(self, sx, sy, sz, s_val, skip_invalid=True):
        imagenum = np.zeros((self.nx, self.ny, self.nz)).astype(np.int64)
        imagesum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)

        return grdf.get_sampled_image(sx, sy, sz, s_val, *self.get_min_and_offset(), imagenum=imagenum, imagesum=imagesum, skip_invalid=skip_invalid)

    def append_sampled_image(self, sx, sy, sz, s_val,
                             imagesum, imagenum,
                             skip_invalid=True):

        if imagenum is None:
            imagenum = np.zeros((self.nx, self.ny, self.nz)).astype(np.int64)
        if imagesum is None:
            imagesum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)
        return get_sampled_image(sx, sy, sz, s_val, *self.get_min_and_offset(), imagenum=imagenum, imagesum=imagesum, skip_invalid=skip_invalid)

    def get_sampled_image2(self, sx, sy, sz, s_val, skip_invalid=True,
                           extent=None):
        imagenum = np.zeros((self.nx, self.ny, self.nz)).astype(np.int64)
        imagesum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)

        #returns imagesum, imagenum
        return grdf.get_sampled_image2(sx, sy, sz, s_val, *self.get_min_and_offset(), imagenum=imagenum, imagesum=imagesum, skip_invalid=skip_invalid, extent=extent)

    def append_sampled_image2(self, sx, sy, sz, s_val,
                              imagesum, imagenum,
                              skip_invalid=True,
                              extent=None
                              ):

        if imagenum is None:
            imagenum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)
        if imagesum is None:
            imagesum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)

        return get_sampled_image2(sx, sy, sz, s_val, *self.get_min_and_offset(), imagenum=imagenum, imagesum=imagesum,
                                  skip_invalid=skip_invalid,
                                  extent=extent)

    def append_sampled_image_inv_dist(self, sx, sy, sz, s_val,
                                      imagesum, imagenum,
                                      skip_invalid=True,
                                      radius=None
                                      ):

        if imagenum is None:
            imagenum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)
        if imagesum is None:
            imagesum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)

        return get_sampled_image_inv_dist(sx, sy, sz, s_val, *self.get_min_and_offset(), imagenum=imagenum, imagesum=imagesum,
                                          skip_invalid=skip_invalid,
                                          radius=radius)
