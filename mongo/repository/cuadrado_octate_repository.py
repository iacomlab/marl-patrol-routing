from mongo_manager.repository.repository_base import RepositoryBase

from mongo.entity.cuadrado_octate import CuadradoMundoOctateReduced


class RepositoryMongoCuadradoOctate(RepositoryBase[CuadradoMundoOctateReduced]):
    def __init__(self) -> None:
        # super().__init__('cuadrado_reduced_copy', CuadradoMundoOctateReduced)
        super().__init__('cuadrado_octates_resumido', CuadradoMundoOctateReduced)

    def find_square_zone(self, zone):
        cuadrados = self.find_many({'zona_algoritmo': str(zone), 'transitable': True}, limit=0)
        x = [h.v5.lat for h in cuadrados]
        y = [h.v5.lon for h in cuadrados]
        return self.find_many(
            {'v5.lat': {'$gte': min(x), '$lte': max(x)}, 'v5.lon': {'$gte': min(y), '$lte': max(y)}}, limit=0)
