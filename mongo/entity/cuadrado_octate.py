from general_utils_j.clases_auxiliares.punto import Punto

from mongo.entity.cuadrado_abstract import CuadradoMundoAbstract
from mongo.entity.cuadricula import Cuadricula


class CuadradoMundoOctateReduced(CuadradoMundoAbstract):


    @staticmethod
    def generar_cuadrado(cuadricula: Cuadricula, p: Punto, geo=True):
        return CuadradoMundoOctateReduced(cuadricula=cuadricula, v5=p, geo=geo)

    def __init__(self, v5: Punto = None, geo: bool = False,
                 distancia_centro: float = 0.5,
                 cuadricula: Cuadricula = None,
                 zona_algoritmo: str = None,
                 vecinos: list = None,
                 transitable: bool = False,
                 vecinos_simulacion: list = None, vecinos_vias: list = None,
                 vias: list = None, vias_ampliadas: dict = None, _id=None, **kwargs):
        super().__init__(cuadricula=cuadricula,
                         v5=v5, geo=geo,
                         distancia_centro=distancia_centro,
                         vecinos=vecinos, _id=_id)
        self.vias = vias
        self.vias_ampliadas = vias_ampliadas
        self.vecinos_simulacion = vecinos_simulacion
        self.vecinos_vias = vecinos_vias
        self.zona_algoritmo = zona_algoritmo
        self.transitable = transitable

    def generar_normales_cuadrados_cuadrados(self, tipologia=None) -> dict:
        general = 0
        general_cuadrados = 0
        general_puntos = 0
        returned = {}
        for x in self.vias:
            if x[1] is not None:
                v = self.vias_ampliadas[str(x[0])]
                td = 0
                for y in v.get('delitos', []):
                    for z, w in y.items():
                        if tipologia is None or z == tipologia:
                            td += w
                td_c = td / v['cuadrados_total']
                td_p = td / v['puntos_total'] * v['puntos_cuadrado']
                general += td
                general_cuadrados += td_c
                general_puntos += td_p
                returned[tuple(x)] = {'delitos': td, 'delitos_cuadrado': td_c, 'delitos_puntos': td_p}
        returned['GENERAL'] = {'delitos': general, 'delitos_cuadrado': general_cuadrados,
                               'delitos_puntos': general_puntos}
        return returned

    def generar_puntuacion_delictiva(self, tipologia=None):
        d = self.generar_normales_cuadrados_cuadrados(tipologia)
        a = [0, 0, 0]
        for x in d.values():
            a[0] += x['delitos']
            a[1] += x['delitos_cuadrado']
            a[2] += x['delitos_puntos']
        return [round(num) for num in a]