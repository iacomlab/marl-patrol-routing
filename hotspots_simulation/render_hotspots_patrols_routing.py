import math
import sys
from math import sqrt

import pygame
from general_utils_j.colors import wcolors
from general_utils_j.patrones.singleton import SingletonMeta
from pygame import DOUBLEBUF

from hotspots_simulation.environment.environment import EnvironmentStreets
from hotspots_simulation.environment.cell_world import CellWorldSimulation

from general_utils_j.umaths import map_value

from hotspots_simulation.transform_hotspots_patrols_routing_pathings import transform_pathing_hotpots_patrol_routing


def close():
    pygame.quit()
    sys.exit(-1)


class RenderHotpotsPatrolRouting(metaclass=SingletonMeta):
    DEFAULT_SIZE = (1000, 1000)

    def __init__(self, world: EnvironmentStreets, render=True, render_size=(1800, 1020),
                 title="Hotspots patrol routing"):
        self._render = render
        self.world = world
        self.render_size = render_size
        self.render_size_inside = (self.render_size[0] - 80, self.render_size[1] - 80)
        if self._render:
            pygame.init()
            self.display = pygame.display.set_mode(self.render_size, DOUBLEBUF)
            self.clock = pygame.time.Clock()
            min_render = min(self.render_size)

            self.font_size = map_value(min_render, 600, 1100,
                                       0, 1, raise_error=True)
            self.radius = map_value(min_render, 600, 1100, 0.3, 1, raise_error=True)

            self.font_size = map_value(self.world.total_cells, 0, 2000,
                                       (35 * self.font_size), -10,
                                       raise_error=False, returned_no_error_min=(40 * self.font_size),
                                       returned_no_error_max=-10)
            self.radius = map_value(self.world.total_cells, 0, 4000,
                                    (16 * self.radius), 0.1,
                                    raise_error=False, returned_no_error_min=(16 * self.radius),
                                    returned_no_error_max=0.1)
            self.distortion_0 = self.render_size_inside[0] / self.DEFAULT_SIZE[0]
            self.distortion_1 = self.render_size_inside[1] / self.DEFAULT_SIZE[1]

            self.a_x_0 = sqrt(pow(self.radius, 2) / (1 + self.distortion_1))
            self.b_x_1 = sqrt(pow(self.radius, 2) / (1 + self.distortion_0))
            if self.font_size is not None:
                self.font = pygame.font.Font(None, int(round(self.font_size, 0)))
            else:
                self.font = None
            self.font_variables = pygame.font.Font(None, 16)
            pygame.display.set_caption(title)
            self.edges = list(world.graph.edges)
            self.colors = [[wcolors.BLACK, wcolors.GREEN, wcolors.RED],
                           [wcolors.GRAY, wcolors.GRAY, wcolors.PURPLE]]

    def _scale_point(self, point):
        return 40 + point[0] * self.distortion_0, 40 + point[1] * self.distortion_1

    def render(self, mode, steps=0, max_steps=0, total_reward=0, reward_step=0):
        if self._render:
            for x in pygame.event.get():
                if x.type == pygame.QUIT:
                    close()
            self.display.fill(wcolors.WHITE)
            pygame.draw.rect(self.display, wcolors.BLACK,
                             [20, 20, self.render_size[0] - 40, self.render_size[1] - 40],
                             2)
            for x in self.edges:
                self.draw_edge(self.world.cells[x[0]], self.world.cells[x[1]])
            for x in self.world.cells.values():
                self.draw_node(x, mode)
            pygame.draw.rect(self.display, wcolors.BLACK, [20, 2, 200, 15], 1)
            pygame.draw.rect(self.display, wcolors.BLACK, [240, 2, 200, 15], 1)
            pygame.draw.rect(self.display, wcolors.BLACK, [460, 2, 200, 15], 1)
            text = self.font_variables.render(f'{steps} / {max_steps}', True, wcolors.BLACK)
            text_2 = self.font_variables.render(str(round(total_reward, 1)), True, wcolors.BLACK)
            text_3 = self.font_variables.render(str(round(reward_step, 1)), True, wcolors.BLACK)
            self.display.blit(text, text.get_rect(center=(110, 8.5)))
            self.display.blit(text_2, text.get_rect(center=(340, 8.5)))
            self.display.blit(text_3, text.get_rect(center=(560, 8.5)))
            pygame.display.flip()
            self.clock.tick(60)
        if steps == max_steps and max_steps > 0:
            transform_pathing_hotpots_patrol_routing(self.world)

    def draw_node(self, cell: CellWorldSimulation, mode):
        position = self._scale_point(cell.position)
        index = 0 if cell.zona_algoritmo == self.world.area else 1
        color = self.colors[index][0]
        text_node = ''
        if mode == 'human_mov':
            color = self.colors[index][1] if cell.visitas_patrols > 0 else color
            text_node = str(round(cell.estimated_crimes, 0))
        elif mode == 'human_off':
            color = self.colors[index][0] if cell.visitas_patrols > 0 else color
            text_node = str(cell.identificador)
        color = self.colors[index][2] if cell.current_patrols_cell > 0 else color
        pygame.draw.circle(self.display, color, position,
                           self.radius, 3 if index == 0 else 2)
        # if self.font is not None:
        #     text = self.font.render(text_node, True, color)
        #     self.display.blit(text, text.get_rect(center=position))

    def draw_edge(self, cell: CellWorldSimulation, cell2: CellWorldSimulation):
        width = 2
        x0, x1 = self._scale_point(cell.position)
        y0, y1 = self._scale_point(cell2.position)
        if x0 < y0:
            if x1 == y1:
                x0 += self.radius - width
                y0 -= self.radius
            else:
                x0 += self.a_x_0 - width / 2
                x1 += self.b_x_1 - width / 2
                y0 -= self.a_x_0
                y1 -= self.b_x_1
        elif x0 == y0:
            x1 += self.radius - width
            y1 -= self.radius
        else:
            x0 -= self.a_x_0 - width / 2
            x1 += self.b_x_1 - width / 2
            y0 += self.a_x_0
            y1 -= self.b_x_1
        i = 0 if cell.zona_algoritmo == self.world.area else 1
        j = 0 if cell2.zona_algoritmo == self.world.area else 1
        color = wcolors.GRAY if i + j > 0 else wcolors.BLACK
        pygame.draw.line(self.display, color, (x0, x1), (y0, y1), width)
        return (x0, x1), (y0, y1), width, color

    def draw_arrow(self, cell: CellWorldSimulation, cell2: CellWorldSimulation, direccion='bi'):
        start, end, width, color = self.draw_edge(cell, cell2)
        rad = math.pi / 180
        trirad = min(max(math.dist(start, end) * 0.2, 2), 10)
        angulo = 160
        color = wcolors.BLACK
        if direccion == 'bi' or direccion is None:
            self._draw_arrow(start, end, angulo, rad, trirad, color, width)
        if direccion == 'bi' or direccion == 'inv':
            self._draw_arrow(end, start, angulo, rad, trirad, color, width)

    def _draw_arrow(self, start, end, angulo, rad, size, color, width):
        rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi / 2
        pygame.draw.line(self.display, color, (end[0], end[1]),
                         (end[0] + size * math.sin(rotation - angulo * rad),
                          end[1] + size * math.cos(rotation - angulo * rad)), width)
        pygame.draw.line(self.display, color, (end[0], end[1]),
                         (end[0] + size * math.sin(rotation + angulo * rad),
                          end[1] + size * math.cos(rotation + angulo * rad)), width)


if __name__ == '__main__':
    from dotenv import load_dotenv
    from utils.db_init import generate_dbs

    from hotspots_simulation.environment.environment import EnvironmentStreets

    load_dotenv()
    generate_dbs()
    zona = 3
    w = EnvironmentStreets(zona, 50, clean_not_accessible=True)
    r = RenderHotpotsPatrolRouting(w, render_size=(1020, 1020))
    while True:
        r.render('human_off')
