# SPDX-FileCopyrightText: 2022 Peter Urban, Ghent University
# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

"""
functions to create echogrids
"""

from numba import njit, prange
import numba.types as ntypes
import numpy as np
import math
import numpy as np

from . import helperfunctions as hlp

@njit
def get_minmax(sx, sy, sz):
    minx = np.nan
    maxx = np.nan
    miny = np.nan
    maxy = np.nan
    minz = np.nan
    maxz = np.nan

    for i in range(len(sx)):
        x = sx[i]
        y = sy[i]
        z = sz[i]

        if not x > minx: minx = x
        if not x < maxx: maxx = x
        if not y > miny: miny = y
        if not y < maxy: maxy = y
        if not z > minz: minz = z
        if not z < maxz: maxz = z

    return minx, maxx, miny, maxy, minz, maxz



@njit
def get_index(val, grd_val_min, grd_res):
    return hlp.round_int((val - grd_val_min) / grd_res)

@njit
def get_index_fraction(val, grd_val_min, grd_res):
    return (val - grd_val_min) / grd_res


@njit
def get_value(index, grd_val_min, grd_res):
    return grd_val_min + grd_res * float(index)


@njit
def get_grd_value(value, grd_val_min, grd_res):
    return get_value(get_index(value, grd_val_min, grd_res), grd_val_min, grd_res)



@njit
def get_index_vals(fraction_index_x : float,
                   fraction_index_y : float,
                   fraction_index_z : float) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):

    """
    Return a vector with fraction and weights for the neighboring grid cells.
    This allows for a linear interpolation (right?)
    :param fraction_index_x: fractional x index (e.g 4.2)
    :param fraction_index_y: fractional y index (e.g 4.2)
    :param fraction_index_z: fractional z index (e.g 4.2)
    :return: - vec X (x indices as int): all indices "touched" by the fractional point
             - vec Y (Y indices as int): all indices "touched" by the fractional point
             - vec Z (Z indices as int): all indices "touched" by the fractional point
             - vec Weights (Weights indices as int): weights
    """

    ifraction_x = fraction_index_x % 1
    ifraction_y = fraction_index_y % 1
    ifraction_z = fraction_index_z % 1

    fraction_x = 1 - ifraction_x
    fraction_y = 1 - ifraction_y
    fraction_z = 1 - ifraction_z

    ix1 = math.floor(fraction_index_x)
    ix2 = math.ceil(fraction_index_x)
    iy1 = math.floor(fraction_index_y)
    iy2 = math.ceil(fraction_index_y)
    iz1 = math.floor(fraction_index_z)
    iz2 = math.ceil(fraction_index_z)

    X = np.array([ix1, ix1, ix1, ix1, ix2, ix2, ix2, ix2])
    Y = np.array([iy1, iy1, iy2, iy2, iy1, iy1, iy2, iy2])
    Z = np.array([iz1, iz2, iz1, iz2, iz1, iz2, iz1, iz2])

    vx = 1 * fraction_x
    vxy = vx * fraction_y
    vxiy = vx * ifraction_y

    vix = 1 * ifraction_x
    vixy = vix * fraction_y
    vixiy = vix * ifraction_y

    WEIGHT = np.array([
        vxy * fraction_z,
        vxy * ifraction_z,
        vxiy * fraction_z,
        vxiy * ifraction_z,
        vixy * fraction_z,
        vixy * ifraction_z,
        vixiy * fraction_z,
        vixiy * ifraction_z
    ])

    return X, Y, Z, WEIGHT


@njit
def get_index_vals2_sup(fraction_index_x_min, fraction_index_x_max):
    ifraction_x_min = fraction_index_x_min % 1
    ifraction_x_max = fraction_index_x_max % 1
    fraction_x_min = 1 - ifraction_x_min
    fraction_x_max = 1 - ifraction_x_max

    if ifraction_x_min < 0.5:
        x1 = int(math.floor(fraction_index_x_min))
        fraction_x1 = 0.5 - ifraction_x_min
    else:
        x1 = int(math.ceil(fraction_index_x_min))
        fraction_x1 = 0.5 + fraction_x_min

    if fraction_x_max >= 0.5:
        x2 = int(math.floor(fraction_index_x_max))
        fraction_x2 = 0.5 + ifraction_x_max
    else:
        x2 = int(math.ceil(fraction_index_x_max))
        fraction_x2 = 0.5 - fraction_x_max

    length = x2 - x1 + 1

    X = np.empty((length)).astype(np.int64)
    W = np.ones((length)).astype(np.float64)

    W[0] = fraction_x1
    W[-1] = fraction_x2

    xm = (x1 + x2) / 2
    xl = (x2 - x1)

    for i, index in enumerate(range(x1, x2 + 1)):
        X[i] = index

    W /= np.sum(W)

    return X, W



# print(sum(WEIGHT))

@njit
def get_index_vals2(fraction_index_x_min, fraction_index_x_max,
                    fraction_index_y_min, fraction_index_y_max,
                    fraction_index_z_min, fraction_index_z_max):
    X_, WX_ = get_index_vals2_sup(fraction_index_x_min, fraction_index_x_max)
    Y_, WY_ = get_index_vals2_sup(fraction_index_y_min, fraction_index_y_max)
    Z_, WZ_ = get_index_vals2_sup(fraction_index_z_min, fraction_index_z_max)

    num_cells = X_.shape[0] * Y_.shape[0] * Z_.shape[0]

    X = np.empty((num_cells)).astype(np.int64)
    Y = np.empty((num_cells)).astype(np.int64)
    Z = np.empty((num_cells)).astype(np.int64)
    W = np.empty((num_cells)).astype(np.float64)

    i = 0
    for x, wx in zip(X_, WX_):
        for y, wy in zip(Y_, WY_):
            for z, wz in zip(Z_, WZ_):
                X[i] = x
                Y[i] = y
                Z[i] = z
                W[i] = wx * wy * wz

                i += 1

    return X, Y, Z, W


@njit()
def get_index_vals_inv_dist(x,xmin,xres,
                            y,ymin,yres,
                            z,zmin,zres,
                            R):
    norm_x = (x - xmin)
    norm_y = (y - ymin)
    norm_z = (z - zmin)

    ix_min = hlp.round_int((norm_x - R ) / xres)
    ix_max = hlp.round_int((norm_x + R ) / xres)
    iy_min = hlp.round_int((norm_y - R ) / yres)
    iy_max = hlp.round_int((norm_y + R ) / yres)
    iz_min = hlp.round_int((norm_z - R ) / zres)
    iz_max = hlp.round_int((norm_z + R ) / zres)


    # X = ntypes.List(ntypes.int64)
    # Y = ntypes.List(ntypes.int64)
    # Z = ntypes.List(ntypes.int64)
    # W = ntypes.List(ntypes.float64)
    X = []
    Y = []
    Z = []
    W = []

    min_dr = R / 10

    for ix in np.arange(ix_min,ix_max):
        dx = norm_x - ix * xres
        dx2 = dx*dx
        for iy in np.arange(iy_min,iy_max):
            dy = norm_y - iy * yres
            dy2 = dy*dy
            for iz in np.arange(iz_min,iz_max):
                dz = norm_z - iz * zres
                dz2 = dz*dz
                dr2 = dx2 + dy2 + dz2
                dr  = math.sqrt(dr2)

                if dr <= R:
                    if dr < min_dr:
                        dr = min_dr
                    X.append(ix)
                    Y.append(iy)
                    Z.append(iz)
                    W.append(1/dr)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    W = np.array(W)

    #W /= np.nansum(W)

    return X, Y, Z, W


@njit()
def get_sampled_image_inv_dist(sx, sy, sz, sv,
                       xmin, xres, nx,
                       ymin, yres, ny,
                       zmin, zres, nz,
                       imagenum,
                       imagesum,
                       radius,
                       skip_invalid=True):

    for i in range(len(sx)):
        x = sx[i]
        y = sy[i]
        z = sz[i]
        v = sv[i]

        if i >= len(radius):
            print(len(radius), len(sx))
            raise RuntimeError('aaaah ')


        IX, IY, IZ, WEIGHT = get_index_vals_inv_dist(x, xmin, xres,
                                                     y, ymin, yres,
                                                     z, zmin, zres,
                                                     radius[i])

        # for ix,iy,iz,w in zip(IX,IY,IZ,WEIGHT):
        for i_ in range(len(IX)):
            ix = int(IX[i_])
            iy = int(IY[i_])
            iz = int(IZ[i_])
            w = WEIGHT[i_]

            if w == 0:
                continue

            if not skip_invalid:
                if ix < 0: ix = 0
                if iy < 0: iy = 0
                if iz < 0: iz = 0

                if abs(ix) >= nx: ix = nx - 1
                if abs(iy) >= ny: iy = ny - 1
                if abs(iz) >= nz: iz = nz - 1
            else:
                if ix < 0: continue
                if iy < 0: continue
                if iz < 0: continue

                if abs(ix) >= nx: continue
                if abs(iy) >= ny: continue
                if abs(iz) >= nz: continue

            # print(ix,iy,iz,v,w)
            if v >= 0:
                imagesum[ix][iy][iz] += v * w
                imagenum[ix][iy][iz] += w


    return imagesum, imagenum


#@njit(parallel = True)
@njit()
def get_sampled_image2(sx, sy, sz, sv,
                       xmin, xres, nx,
                       ymin, yres, ny,
                       zmin, zres, nz,
                       imagenum,
                       imagesum,
                       extent=None,
                       skip_invalid=True):


    for i in range(len(sx)):
        x = sx[i]
        y = sy[i]
        z = sz[i]
        v = sv[i]

        if extent is None:
            IX, IY, IZ, WEIGHT = get_index_vals(
                get_index_fraction(x, xmin, xres),
                get_index_fraction(y, ymin, yres),
                get_index_fraction(z, zmin, zres)
            )
        else:
            if i >= len(extent):
                print(len(extent), len(sx))
                raise RuntimeError('aaaah ')

            length_2 = extent[i] / 2

            IX, IY, IZ, WEIGHT = get_index_vals2(
                get_index_fraction(x - length_2, xmin, xres), get_index_fraction(x + length_2, xmin, xres),
                get_index_fraction(y - length_2, ymin, yres), get_index_fraction(y + length_2, ymin, yres),
                get_index_fraction(z - length_2, zmin, zres), get_index_fraction(z + length_2, zmin, zres)
            )

        # for ix,iy,iz,w in zip(IX,IY,IZ,WEIGHT):
        for i_ in range(len(IX)):
            ix = int(IX[i_])
            iy = int(IY[i_])
            iz = int(IZ[i_])
            w = WEIGHT[i_]

            if w == 0:
                continue

            if not skip_invalid:
                if ix < 0: ix = 0
                if iy < 0: iy = 0
                if iz < 0: iz = 0

                if abs(ix) >= nx: ix = nx - 1
                if abs(iy) >= ny: iy = ny - 1
                if abs(iz) >= nz: iz = nz - 1
            else:
                if ix < 0: continue
                if iy < 0: continue
                if iz < 0: continue

                if abs(ix) >= nx: continue
                if abs(iy) >= ny: continue
                if abs(iz) >= nz: continue

            # print(ix,iy,iz,v,w)
            if v >= 0:
                imagesum[ix][iy][iz] += v * w
                imagenum[ix][iy][iz] += w


    return imagesum, imagenum


@njit
def get_sampled_image(sx, sy, sz, sv,
                      xmin, xres, nx,
                      ymin, yres, ny,
                      zmin, zres, nz,
                      imagenum,
                      imagesum,
                      skip_invalid = True):


    for i in range(len(sx)):
        x = sx[i]
        y = sy[i]
        z = sz[i]
        v = sv[i]

        ix = get_index(x, xmin, xres)
        iy = get_index(y, ymin, yres)
        iz = get_index(z, zmin, zres)

        if not skip_invalid:
            if ix < 0: ix = 0
            if iy < 0: iy = 0
            if iz < 0: iz = 0

            if abs(ix) >= nx: ix = nx - 1
            if abs(iy) >= ny: iy = ny - 1
            if abs(iz) >= nz: iz = nz - 1
        else:
            if ix < 0: continue
            if iy < 0: continue
            if iz < 0: continue

            if abs(ix) >= nx: continue
            if abs(iy) >= ny: continue
            if abs(iz) >= nz: continue

        if v >= 0:
            imagesum[ix][iy][iz] += v
            imagenum[ix][iy][iz] += 1

    return imagesum, imagenum

class GRIDDER(object):

    def __init__(self, xres, yres, zres,
                 min_x, max_x,
                 min_y, max_y,
                 min_z, max_z,
                 xbase=None,
                 ybase=None,
                 zbase=None):

        # resolution in meter
        self.xres = xres
        self.yres = yres
        self.zres = zres

        if False:
            if xbase is None:
                self.xbase = self.xres * 0.5
            else:
                self.xbase = xbase
            if ybase is None:
                self.ybase = self.yres * 0.5
            else:
                self.ybase = ybase
            if zbase is None:
                self.zbase = self.zres * 0.5
            else:
                self.zbase = zbase

        else:
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
        self.nx = math.floor(round(((self.xmax - self.xmin) / self.xres), 8)) + 1  # num of elements x
        self.ny = math.floor(round(((self.ymax - self.ymin) / self.yres), 8)) + 1  # num of elements y
        self.nz = math.floor(round(((self.zmax - self.zmin) / self.zres), 8)) + 1  # num of elements z
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

    def get_sampled_image(self,sx, sy, sz, s_val,skip_invalid = True):
        #returns imagesum, imagenum
        return get_sampled_image(sx, sy, sz, s_val, *self.get_min_and_offset(),skip_invalid = skip_invalid)

    def append_sampled_image(self,sx, sy, sz, s_val,
                             imagesum, imagenum,
                             skip_invalid = True):

        if imagenum is None:
            imagenum = np.zeros((self.nx, self.ny, self.nz)).astype(np.int64)
        if imagesum is None:
            imagesum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)
        return get_sampled_image(sx, sy, sz, s_val, *self.get_min_and_offset(),imagenum = imagenum, imagesum = imagesum,skip_invalid = skip_invalid)

    def get_sampled_image2(self,sx, sy, sz, s_val,skip_invalid = True,
                          extent   = None):
        #returns imagesum, imagenum
        return get_sampled_image2(sx, sy, sz, s_val, *self.get_min_and_offset(),skip_invalid = skip_invalid,extent=extent)

    def append_sampled_image2(self,sx, sy, sz, s_val,
                             imagesum, imagenum,
                             skip_invalid = True,
                          extent   = None
                              ):

        if imagenum is None:
            imagenum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)
        if imagesum is None:
            imagesum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)

        return get_sampled_image2(sx, sy, sz, s_val, *self.get_min_and_offset(), imagenum=imagenum, imagesum=imagesum,
                                  skip_invalid=skip_invalid,
                                  extent=extent)

    def append_sampled_image_inv_dist(self,sx, sy, sz, s_val,
                             imagesum, imagenum,
                             skip_invalid = True,
                          radius   = None
                              ):

        if imagenum is None:
            imagenum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)
        if imagesum is None:
            imagesum = np.zeros((self.nx, self.ny, self.nz)).astype(np.float64)

        return get_sampled_image_inv_dist(sx, sy, sz, s_val, *self.get_min_and_offset(), imagenum=imagenum, imagesum=imagesum,
                                  skip_invalid=skip_invalid,
                                  radius=radius)

if __name__ == '__main__':
    minx = -22
    maxx =  22
    miny = -22
    maxy =  22
    minz = -120
    maxz = 0

    xres = 1
    yres = 1
    zres = 1

    gridder = GRIDDER(xres, yres, zres,
                      minx, maxx,
                      miny, maxy,
                      minz, maxz,
                      xbase=0.5)

    print(minx, gridder.get_x_grd_value(minx), gridder.get_x_index(minx))
    print(maxx, gridder.get_x_grd_value(maxx), gridder.get_x_index(maxx))
    print()
    print(xres, gridder.get_x_value(1) - gridder.get_x_value(0))
    print()
    index = 10
    print(index, gridder.get_x_grd_value(index))

    print(gridder.get_extent_x())

