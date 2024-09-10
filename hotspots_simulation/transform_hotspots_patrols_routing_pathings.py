import random
from copy import copy

import pandas as pd

from hotspots_simulation.environment.environment import EnvironmentStreets
from mongo.entity.cuadrado_octate import CuadradoMundoOctateReduced


def get_same_roads(square_1: CuadradoMundoOctateReduced, square_2: CuadradoMundoOctateReduced):
    k = [x for x in square_1.vias if x in square_2.vias]
    k_1 = [y for y in k if y[1] is not None]
    return k_1 if len(k_1) > 0 else k


def return_minimal_roads(vias) -> list:
    returned_vias = []
    cont = 0
    for i, x in enumerate(vias):
        if cont > 0:
            cont -= 1
            continue
        c = None
        if type(x[0]) is not list:
            returned_vias.append(x)
            continue
        if i == len(vias) - 1:
            returned_vias.append(x[0])  # choose best one
        for y in x:
            cont_interno = 0
            for z in vias[i + 1:]:
                if y in z or y == z:
                    cont_interno += 1
                else:
                    break
            if cont_interno > cont:
                cont = cont_interno
                c = y
            elif cont_interno == cont:
                pass  # choose best one
        returned_vias.append(c if c is not None else x[0])  # choose best one
    return returned_vias


def reduce_pathing_2(vias):
    returned_vias = []
    k = 0
    for i, x in enumerate(vias):
        if k > 0:
            k = k - 1
            continue
        if type(x[0]) is not list:
            returned_vias.append(x)
        else:
            if i + 1 == len(vias):
                # Meter mas delitos
                continue
            if type(vias[i + 1][0]) is not list:
                if vias[i + 1] in x:
                    returned_vias.append(vias[i + 1])
                    k = 1
                else:
                    # Meter mas delitos
                    pass
                continue
            p = 1
            for y in vias[i + 1:]:
                if type(y[0]) is list:
                    p += 1
                else:
                    p += 1
                    break
            k = p - 1
            returned_vias = returned_vias + return_minimal_roads(copy(vias[i:(i + p)]))
    return returned_vias


def reduce_pathing_1(vias):
    k = [x[0] if len(x) == 1 else x for x in vias]
    j = []
    l = None
    for x in k:
        if type(x[0]) is list:
            if l in x:
                continue
            l = None
            j.append(x)
        elif x != l:
            l = x
            j.append(x)

    return reduce_pathing_2(j)

count = 0
def transform_pathing_hotpots_patrol_routing(world: EnvironmentStreets):
    world = world
    id_square_pathings_dict = world.get_paths()
    cuadrados = {x.v5.get_tuple(): x for x in world.get_cuadrados_environment()}
    square_pathings_dict = {x: [cuadrados[world.identificador_v5[z]] for z in y]
                            for x, y in id_square_pathings_dict.items()}
    vias = {
        x:
            [get_same_roads(y[z], y[z + 1]) for z in range(0, len(y) - 1)]
        for x, y in square_pathings_dict.items()
    }
    vias_2 = {
        x:
            reduce_pathing_1(y)
        for x, y in vias.items()
    }

    p = []
    c = []
    i = []
    for x, y in square_pathings_dict.items():
        for z in y:
            p.append(x.agent_id)
            cell_v5 = z.v5.get_tuple()
            c.append(cell_v5)
            i.append(world.nodes_edges_v5[cell_v5])
    pd.DataFrame.from_dict({'patrol': p, 'v5_tuple': c, 'identificador': i}).to_csv(
        f'test_{random.randint(0, 100_000)}.csv', sep=';', index=False)
    global count
    count += 1
    if count >= 100:
        import ray
        ray.shutdown()
    return vias_2
