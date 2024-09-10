import abc

from general_utils_j.clases_auxiliares.punto import Punto
from mongo_manager.entity.objeto_mongo_abstract import ObjetoMongoAbstract

from mongo.entity.cuadricula import Cuadricula


class CuadradoMundoAbstract(ObjetoMongoAbstract):

    #   V1 --------- V2
    #   |             |
    #   |     V5      |
    #   |             |
    #   V3 --------- V4

    def __init__(self, cuadricula: Cuadricula, v5: Punto, geo: bool = False, distancia_centro: float = 0.5,
                 vecinos: list = None, _id=None,
                 **kwargs):
        super().__init__(_id, **kwargs)
        self.cuadricula = cuadricula
        if geo:
            v5 = Punto(
                self.cuadricula.get_transformada_lat(v5.lat),
                self.cuadricula.get_transformada_lon(v5.lon)
            )
        self.v5 = v5
        self.distancia_centro = distancia_centro
        if vecinos is None:
            vecinos = cuadricula.generar_vecinos(v5)
        self.vecinos = vecinos

        lat = self.v5.lat
        lon = self.v5.lon

        self.v1 = Punto(lat - self.distancia_centro, lon - self.distancia_centro)
        self.v2 = Punto(lat - self.distancia_centro, lon + self.distancia_centro)
        self.v3 = Punto(lat + self.distancia_centro, lon - self.distancia_centro)
        self.v4 = Punto(lat + self.distancia_centro, lon + self.distancia_centro)

    @property
    def identificador(self):
        return self.v5.get_tuple()

    def get_dict(self, id_mongo=True, id_as_string=False) -> dict:
        d = super().get_dict(id_mongo, id_as_string)
        d['v5'] = {'lat': self.v5.lat, 'lon': self.v5.lon}
        d.pop('v1')
        d.pop('v2')
        d.pop('v3')
        d.pop('v4')
        return d

    def get_cuadrados_vecinos(self):
        return [type(self)(v5=Punto(*x), cuadricula=self.cuadricula) for x in self.vecinos]

    @staticmethod
    def prepare_dict_for_generated_object(dictionary: dict, attr: dict) -> dict:
        dicctionary = ObjetoMongoAbstract.prepare_dict_for_generated_object(dictionary, attr)
        v5 = dictionary.get('v5', None)
        dicctionary['v5'] = Punto(v5.get('lat'), v5.get('lon'))
        return dicctionary

    @staticmethod
    @abc.abstractmethod
    def generar_cuadrado(cuadricula: Cuadricula, p: Punto, geo=True):
        pass

    def devolver_coordenadas_geo(self):
        return Punto(self.cuadricula.get_transformada_lat_inv(self.v1.lat),
                     self.cuadricula.get_transformada_lon_inv(self.v1.lon)), \
               Punto(self.cuadricula.get_transformada_lat_inv(self.v2.lat),
                     self.cuadricula.get_transformada_lon_inv(self.v2.lon)), \
               Punto(self.cuadricula.get_transformada_lat_inv(self.v3.lat),
                     self.cuadricula.get_transformada_lon_inv(self.v3.lon)), \
               Punto(self.cuadricula.get_transformada_lat_inv(self.v4.lat),
                     self.cuadricula.get_transformada_lon_inv(self.v4.lon)), \
               Punto(self.cuadricula.get_transformada_lat_inv(self.v5.lat),
                     self.cuadricula.get_transformada_lon_inv(self.v5.lon))

    @property
    def posicion_real(self):
        return (self.cuadricula.get_transformada_lat_inv(self.v5.lat),
                self.cuadricula.get_transformada_lon_inv(self.v5.lon))

    def devolver_coordenadas_mundo(self):
        return self.v1, self.v2, self.v3, self.v4, self.v5

    def pintar_cuadrado(self, geo=True):
        if geo:
            print(self.__pintar_cuadrado_str(*self.devolver_coordenadas_geo()))
        else:
            print(self.__pintar_cuadrado_str(*self.devolver_coordenadas_mundo()))

    @staticmethod
    def __pintar_cuadrado_str(v1, v2, v3, v4, v5):
        return '\n\tV1 --------- V2\n\t|             |\n\t|     V5      |\n\t|             |\n\tV3 --------- V4\n' + \
               "\nV1: {} \nV2: {} \nV3: {} \nV4: {} \nV5: {}".format(v1, v2, v3, v4, v5)

    def __repr__(self):
        return "Cuadrado: V5: {}".format(self.v5)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self.v5.lat == other.v5.lat and self.v5.lon == other.v5.lon and self.cuadricula == other.cuadricula

    def __hash__(self):
        return hash(self.v5.lat) ^ hash(self.v5.lat) ^ hash(type(self)) ^ hash(self.cuadricula)

    def __lt__(self, other):
        return self.posicion_real[0] < other.posicion_real[0] or \
               (self.posicion_real[0] == other.posicion_real[0] and self.posicion_real[1] < other.posicion_real[1])

    @staticmethod
    def sort_print(lista) -> list:
        return sorted(lista, key=lambda x: (-x.v5.lat, x.v5.lon))

    @staticmethod
    def get_attr_nested_objects() -> dict:
        return {'cuadricula': Cuadricula}
