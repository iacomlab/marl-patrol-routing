import math

from general_utils_j.clases_auxiliares.punto import Punto
from mongo_manager.entity.objeto_mongo_abstract import ObjetoMongoAbstract


class Cuadricula(ObjetoMongoAbstract):

    def __init__(self, numero_celdas_anchura, numero_celdas_altura,
                 lat_min=36.6440969, lat_max=36.789936,
                 lon_min=-4.576464, lon_max=-4.289452, **kwargs):
        super().__init__(None, **kwargs)
        self.numero_celdas_anchura = numero_celdas_anchura
        self.numero_celdas_altura = numero_celdas_altura

        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

        self.extension_lat = (self.lat_max - self.lat_min)
        self.extension_lon = (self.lon_max - self.lon_min)

        self.lat_celda = self.extension_lat / self.numero_celdas_altura
        self.lon_celda = self.extension_lon / self.numero_celdas_anchura

    def get_transformada_lat(self, x):
        if x > self.lat_max or x < self.lat_min:
            print("Hace falta extender el mapa y {}".format(x))
            return math.inf
        return math.trunc((x - self.lat_min) * self.numero_celdas_altura / self.extension_lat)

    def get_transformada_lon(self, x):
        if x > self.lon_max or x < self.lon_min:
            print("Hace falta extender el mapa x {}".format(x))
            return math.inf
        return math.trunc((x - self.lon_min) * self.numero_celdas_anchura / self.extension_lon)

    def get_transformada_lat_inv(self, x, rounded=6):
        if x >= self.numero_celdas_altura or x < 0:
            print("Punto no presente en el mapa {}".format(x))
            return 0
        h = self.lat_celda * (x + 0.5) + self.lat_min
        return round(h, rounded)

    def get_transformada_lon_inv(self, x, rounded=6):
        if x >= self.numero_celdas_anchura or x < 0:
            print("Punto no presente en el mapa x {}".format(x))
            return 0
        h = self.lon_celda * (x + 0.5) + self.lon_min
        return round(h, rounded)

    def generar_vecinos(self, v5: Punto):
        lat_max = v5.lat + 1
        lat_min = v5.lat - 1
        lon_max = v5.lon + 1
        lon_min = v5.lon - 1
        vecinos = []
        if lat_max < self.numero_celdas_altura - 1:
            if lon_min > 0:
                vecinos.append((lat_max, lon_min))
            if lon_max < self.numero_celdas_altura - 1:
                vecinos.append((lat_max, lon_max))
            vecinos.append((lat_max, v5.lon))

        if lat_min > 0:
            if lon_min > 0:
                vecinos.append((lat_min, lon_min))
            if lon_max < self.numero_celdas_anchura - 1:
                vecinos.append((lat_min, lon_max))
            vecinos.append((lat_min, v5.lon))

        if lon_min > 0:
            vecinos.append((v5.lat, lon_min))
        if lon_max < self.numero_celdas_anchura - 1:
            vecinos.append((v5.lat, lon_max))
        return vecinos
