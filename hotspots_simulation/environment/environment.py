import math
import random

import numpy as np
from general_utils_j.umaths import map_value
from networkx import Graph
import networkx as nx
import matplotlib
import pylab
import logging

from hotspots_simulation.agents import AbstractAgent
from hotspots_simulation.environment.cell_world import CellWorld, CellWorldSimulation
from mongo import RepositoryMongoCuadradoOctate
from mongo.entity.cuadrado_octate import CuadradoMundoOctateReduced

matplotlib.use('Agg')
import matplotlib.backends.backend_agg as agg


def get_direccion_v5(v5_o, v5_d) -> int:
    if v5_o[0] < v5_d[0]:
        if v5_o[1] > v5_d[1]:
            return 0  # Arriba Izquierda
        elif v5_o[1] == v5_d[1]:
            return 1  # Arriba
        else:
            return 2  # Arriba Derecha
    elif v5_o[0] == v5_d[0]:
        if v5_o[1] > v5_d[1]:
            return 3  # Izquierda
        elif v5_o[1] == v5_d[1]:
            return 4  # No moverse
        else:
            return 5  # Derecha
    else:
        if v5_o[1] > v5_d[1]:
            return 6  # Abajo Izquierda
        elif v5_o[1] == v5_d[1]:
            return 7  # Abajo
        else:
            return 8  # Abajo Derecha


class EnvironmentStreets:

    def __init__(self, area=2, steps=200, clean_not_accessible=True, initial_position='random'):
        self.area = str(area)
        self.initial_position = initial_position
        self.clean_not_accessible = clean_not_accessible

        cuadrados_clean = self.get_cuadrados_environment(clean_not_accessible)
        self.nodes_edges_v5 = {v.v5.get_tuple(): k for k, v in
                               enumerate(cuadrados_clean)}
        self.identificador_v5 = {v: k for k, v in self.nodes_edges_v5.items()}
        x0, x1 = [x[0] for x in self.nodes_edges_v5], [x[1] for x in self.nodes_edges_v5]
        self.min_edges = [min(x0), min(x1)]
        self.max_edges = [max(x0), max(x1)]

        nodes_edges = {x.v5.get_tuple(): [y for y in x.vecinos_vias] for x in cuadrados_clean}

        self.graph = Graph()
        for x, y in self.nodes_edges_v5.items():
            self.graph.add_node(y, v5=x)
        for x, y in nodes_edges.items():
            for z in y:
                k = self.nodes_edges_v5.get(tuple(z), None)
                if k:
                    self.graph.add_edge(self.nodes_edges_v5[x], k)

        self.agents = []
        self.cells: dict[int, CellWorldSimulation] = {}

        total_offenses = sum([x.generar_puntuacion_delictiva()[2]
                              for x in cuadrados_clean])
        for x in cuadrados_clean:
            identificador = self.nodes_edges_v5[x.v5.get_tuple()]
            self.cells[identificador] = self.generate_cell_world(x, identificador,
                                                                 self.get_neightbours_directions(identificador),
                                                                 self.nodes_edges_v5,
                                                                 total_offenses, steps)
        self.total_estimated_crimes = sum(
            x.estimated_crimes if x.zona_algoritmo == self.area else 0 for x in self.cells.values())
        self.total_cells = len(self.cells)
        self.areas_obs = [0 if self.area != x.zona_algoritmo else 1 for x in self.cells.values()]

    def get_cuadrados_environment(self, clean_not_accesible=True):
        repo = RepositoryMongoCuadradoOctate()
        cuadrados = repo.find_square_zone(self.area)
        dict_cuadrados = {x.v5.get_tuple(): x for x in cuadrados}
        logging.log(level=logging.INFO, msg=f'Cuadrados conseguidos de la base de datos: {len(cuadrados)}')
        if clean_not_accesible:
            cuadrados_clean = []
            for x in cuadrados:
                if not (not x.transitable or len(x.vecinos_vias) == 0):
                    for y in x.vecinos_vias:
                        if dict_cuadrados.get(tuple(y), None) is not None:
                            cuadrados_clean.append(x)
                            break
        else:
            cuadrados_clean = cuadrados
        logging.log(level=logging.INFO, msg=f'Cuadrados conseguidos de la base de datos: {len(cuadrados_clean)}')
        return CuadradoMundoOctateReduced.sort_print(cuadrados_clean)

    def reset_cells(self):
        for x in self.cells.values():
            x.reset()

    def get_block_cell_full(self, stat):
        vecinos = [[-2] * (self.max_edges[1] + 1 - self.min_edges[1]) for _ in
                   range(0, 1 + (self.max_edges[0] - self.min_edges[0]))]
        for x, x_1 in enumerate(range(self.min_edges[0], self.max_edges[0] + 1)):
            for y, y_1 in enumerate(range(self.min_edges[1], self.max_edges[1] + 1)):
                identificador = self.nodes_edges_v5.get((x_1, y_1), -2)
                if identificador != -2:
                    c = self.cells[identificador]
                    if stat == 'crimes':
                        vecinos[x][y] = round(
                            c.estimated_crimes / max(c.visitas_patrols, 1)) if c.zona_algoritmo == self.area else -1
                    elif stat == 'visits':
                        vecinos[x][y] = c.visitas_patrols if c.zona_algoritmo == self.area else -1
        return np.array(vecinos, dtype='float64').flatten()

    def get_block_cells(self, cell: CellWorldSimulation, *, l_sight=5, stat: str = None):
        if l_sight < 0:
            return self.get_block_cell_full(stat)
        size_square = l_sight * 2 + 1
        vecinos = [[-2] * size_square for _ in range(0, size_square)]
        for x, y in enumerate(list(range(cell.v5[0] - l_sight, cell.v5[0] + l_sight + 1))[::-1]):
            for z, l in enumerate(range(cell.v5[1] - l_sight, cell.v5[1] + l_sight + 1)):
                identificador = self.nodes_edges_v5.get((y, l), -2)
                if identificador != -2:
                    cell_i = self.cells[identificador]
                    if not stat:
                        vecinos[x][z] = identificador
                        continue
                    if cell_i.zona_algoritmo != self.area:
                        vecinos[x][z] = -1
                        continue
                    else:
                        if stat == 'crimes':
                            vecinos[x][z] = round(cell_i.estimated_crimes / max(cell_i.visitas_patrols, 1))
                        elif stat == 'visits':
                            vecinos[x][z] = cell_i.visitas_patrols
        return np.array(vecinos, dtype='float64').flatten()

    def seed(self, seed=None):
        random.seed(seed)

    def initialize_new_agent(self, agent):
        self.agents.append(agent)
        agent.set_init_cell(self.cells[random.sample(self.cells.keys(), 1)[0]])

    def destroy_agent(self, agent: AbstractAgent):
        self.agents.remove(agent)
        agent.delete_agent()

    def randomize_agents_possitions(self):
        c = list(filter(lambda f: f.zona_algoritmo == self.area, self.cells.values()))
        chosed = []
        if self.initial_position == 'random':
            for i, x in enumerate(self.agents):
                choose = random.sample(c, 1)[0]
                chosed.append(choose)
                x.set_init_cell(choose)
        elif self.initial_position == 'best':
            c_s = sorted(c, key=lambda f: f.estimated_crimes, reverse=True)[:30]
            for i, x in enumerate(self.agents):
                choose = c_s[i]
                chosed.append(choose)
                x.set_init_cell(choose)
        elif self.initial_position == 'hybrid':
            c_s = sorted(c, key=lambda f: f.estimated_crimes, reverse=False)[:30]
            for i, x in enumerate(self.agents):
                choose = None
                while choose is None or choose in chosed:
                    choose = random.sample(c_s, 1)[0]
                chosed.append(choose)
                x.set_init_cell(choose)


    def move_to(self, a, direction):
        cell_des = a.cell.vecinos_actions[direction]
        if math.isnan(cell_des):
            return
        a.change_cell(self.cells[int(cell_des)])

    def get_position_render(self, v5):
        render = 1000
        return (
            round(map_value(v5[1], in_min=self.min_edges[1], in_max=self.max_edges[1],
                            out_min=0, out_max=render), 0),
            round(map_value(v5[0], in_min=self.min_edges[0], in_max=self.max_edges[0],
                            out_min=render, out_max=0), 0)
        )

    def generate_cell_world(self, c: CuadradoMundoOctateReduced, identificador,
                            neightbours_directions, nodes_edges_v5,
                            total_offenses, steps):
        v5 = c.v5.get_tuple()
        identificador = identificador
        vecinos_actions = np.empty(9)
        vecinos_actions[:] = np.nan
        masking = np.zeros(9)
        vecinos_v5 = []
        vecinos_identificadores = []
        for i in neightbours_directions:
            vecinos_identificadores.append(nodes_edges_v5[i[1]])
            vecinos_v5.append(i[1])
            vecinos_actions[i[2]] = nodes_edges_v5[i[1]]
            masking[i[2]] = 1
        crime_rate = c.generar_puntuacion_delictiva()[2] / total_offenses
        return CellWorldSimulation(CellWorld(v5=v5, zona_algoritmo=c.zona_algoritmo,
                                             identificador=identificador,
                                             real_crimes=c.generar_puntuacion_delictiva()[2],
                                             crime_rate=crime_rate,
                                             estimated_crimes=crime_rate * 100 * steps,
                                             vecinos_identificadores=vecinos_identificadores,
                                             vecinos_v5=vecinos_v5,
                                             vecinos_actions=vecinos_actions,
                                             vecinos_mask=masking,
                                             position=self.get_position_render(v5)))

    def get_neightbours_directions(self, node):
        v_o = self.identificador_v5[node]
        returned = [(node, v_o, 4)]
        for x in self.graph.neighbors(node):
            v_d = self.identificador_v5[x]
            a = get_direccion_v5(v_o, v_d)
            returned.append((x, v_d, a))
        return returned

    def reset(self):
        # Borrar ventana y reiniciar steps alive
        for a in self.agents:
            a.reset_memory()
        # Resetear la parte estatica de las celdas
        self.reset_cells()
        # Randomizar la posicion de cada agente
        self.randomize_agents_possitions()

    def _generate_figure(self, node_colors: list = None, labels: dict = None, **more_options):
        node_colors = node_colors if node_colors is not None else 'black'
        options = {
            'font_size': 5,
            'node_size': 50,
            'node_color': node_colors,
            'alpha': 0.4,
            'edge_color': 'black',
            'labels': {y: str(y) for x, y in self.nodes_edges_v5.items()} if labels is None else labels
        }
        pylab.close('all')
        opt = {**options, **more_options}
        pos = {
            y: [map_value(x[1], in_min=self.min_edges[1], in_max=self.max_edges[1], out_min=-0.5, out_max=0.9),
                map_value(x[0], in_min=self.min_edges[0], in_max=self.max_edges[0], out_min=-0.5, out_max=0.9)]
            for x, y in self.nodes_edges_v5.items()}
        f = pylab.figure(figsize=(10, 10), dpi=100)
        nx.draw_networkx(self.graph, pos, **opt)
        return agg.FigureCanvasAgg(f)

    def get_image(self, node_colors: list = None, labels: dict = None, **more_options):
        return self._generate_figure(node_colors=node_colors, labels=labels, **more_options)

    def get_cells_visited(self):
        return [c.visitas_patrols for c in self.cells.values()]

    def get_cells_estimated_cells(self):
        return [round(c.estimated_crimes / max(c.visitas_patrols, 1), 3) if c.zona_algoritmo == self.area else -1 for c
                in self.cells.values()]

    def get_paths(self):
        return {patrol: patrol.memory_cell.memory_cells for patrol in self.agents}

    def get_cells_area(self):
        return list(filter(lambda x: x.zona_algoritmo == self.area, self.cells.values()))

    def get_pai_star_cells(self, por: float = 3.0, print_number=False):
        d = sorted(filter(lambda x: x.zona_algoritmo == self.area, self.cells.values()),
                   key=lambda x: x.estimated_crimes, reverse=True)
        f = math.floor(len(d) * por / 100)
        if print_number:
            print(f)
        return d[:f]

    def get_pai_star(self, por: float = 3.0):
        d = sorted(filter(lambda x: x.zona_algoritmo == self.area, self.cells.values()),
                   key=lambda x: x.estimated_crimes, reverse=True)
        f = math.floor(len(d) * por / 100)
        return (sum([x.estimated_crimes for x in d[:f]]) / sum([x.estimated_crimes for x in d])) / (f / len(d))


if __name__ == '__main__':
    from dotenv import load_dotenv
    from utils.db_init import generate_dbs

    # import tracemalloc

    # tracemalloc.start()

    load_dotenv()
    generate_dbs()
    zona = 3
    w = EnvironmentStreets(zona, 50)
    print(w.cells[0].identificador)
    print(w.get_block_cells(w.cells[216], l_sight=2, stat=''))
    # print(w.get_pai_star(5))
    # print([x.identificador for x in w.get_pai_star_cells(5)])
