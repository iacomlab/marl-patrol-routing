import math

from pyproj import Proj

# __myProj = Proj("+proj=utm +zone=30N, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
__myProj = Proj(proj='utm', zone=30, ellps='WGS84', units='m', datum='WGS84')

import pyproj

def metros_lat_lon(x):
    lon, lat = __myProj(x[0], x[1], inverse=True)
    return lat, lon


def lat_lon_metros(x):
    lon, lat = __myProj(x[0], x[1], inverse=False)
    return lat, lon


def get_transformada_lat(x):
    if x > 36.789035888 or x < 36.644997513:
        print("Hace falta extender el mapa y {}".format(x))
        return math.inf
    return ((x - 36.644997513) * 80) / (36.789035888 - 36.644997513) - 40


def get_transformada_lon(x):
    if x > -4.290563814 or x < -4.575351230:
        print("Hace falta extender el mapa x {}".format(x))
        return math.inf
    return ((x + 4.575351230) * 128) / (-4.290563814 + 4.575351230) - 64


def get_transformada_lat_inv(x):
    if x > 41 or x < -41:
        print("Punto no presente en el mapa {}".format(x))
        return 0
    return ((x + 40) * (36.789035888 - 36.644997513)) / 80 + 36.644997513


def get_transformada_lon_inv(x):
    if x > 65 or x < -65:
        print("Punto no presente en el mapa x {}".format(x))
        return 0
    return ((x + 64) * (-4.290563814 + 4.575351230)) / 128 - 4.575351230


if __name__ == '__main__':
    wgs84_ellipse = pyproj.crs.CRS.from_epsg(4979)
    wgs84_msl = pyproj.crs.CRS.from_epsg(9705)
    tform = pyproj.transformer.Transformer.from_crs(crs_from=wgs84_ellipse, crs_to=wgs84_msl, always_xy=True)
    t = pyproj.transformer.Transformer.from_crs("epsg:4979", "epsg:4326+3855", always_xy=True)
    # t.transform(9.88, 54.44, 100)
    # g = __myProj(-4.4292374, 36.7167616, inverse=False)
    print(t.transform(36.7167616, -4.4292374, 12))
