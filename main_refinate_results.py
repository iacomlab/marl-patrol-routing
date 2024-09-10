import math
import os
from collections import defaultdict
import pandas as pd

from hotspots_simulation.environment.environment import EnvironmentStreets

path_vdppo = 'exp_results/vdppo_gru_hotspots_patrols_routing'
path_ippo = 'exp_results/ippo_gru_hotspots_patrols_routing'
path_mappo = 'exp_results/mappo_gru_hotspots_patrols_routing'

conf_vd = {
    'vd1': 'VDPPO line of sight 1',
    'vd3': 'VDPPO line of sight 3',
    'vd6': 'VDPPO line of sight 6',
}

def counter_cell_routes(routes):
    d = defaultdict(int)
    for x in routes:
        for y in x:
            d[y] += 1
    return d


def get_pn(cell, counter, patrols, n_steps=50):
    return counter.get(str(cell.identificador), 0) / (100 * n_steps * patrols)


def calculate_entropy(w: EnvironmentStreets, patrols, route_list, n_steps=50):
    t = 0
    counter = counter_cell_routes(route_list)
    for x in w.get_cells_area():
        pn = get_pn(x, counter, patrols, n_steps)
        if pn == 0:
            continue
        k = pn * math.log(pn)
        t += k
    return -t


def calculate_pai(w, number, results, simulations):
    pais = {x: w.get_pai_star_cells(x) for x in number}
    resultados_calculo = {}
    for x, pai_cells in pais.items():
        porcentaje = 0
        for y in pai_cells:
            h = results[y.identificador] / simulations
            porcentaje += h * 100
        resultados_calculo[x] = round(porcentaje / (100 * len(pai_cells)), 3)
    return resultados_calculo


def group_words_by_prefix(words):
    groups = {}
    for word in words:
        numbers = word.split(',')
        for i in range(len(numbers)):
            prefix = ','.join(numbers[:i + 1])
            if prefix in groups:
                groups[prefix] += 1
            else:
                groups[prefix] = 1
    return {prefix: count for prefix, count in groups.items() if count > 1 and len(prefix.split(",")) > 20}


def calculate_crimes(w, resultsss):
    delitos = 0
    for re in resultsss:
        for x, y in re.items():
            i = 1
            cell = w.cells[x]
            while i <= y:
                delitos += cell.estimated_crimes / i
                i += 1
    return delitos / len(resultsss)


def main():
    area = 9
    initial = 'best'
    simulations = 100
    w = EnvironmentStreets(area, 50, clean_not_accessible=True)
    print(len(w.cells))

    path = path_vdppo + f'{area}/{initial}'
    conf = conf_vd
    start = 'VDPPO'
    resultss = {}
    resultss2 = {}
    routes_model = {}
    for patrols in ['2_patrullas', '5_patrullas', '10_patrullas']:
        path1 = path + '/' + patrols + '/'
        for filename in os.listdir(path1):
            if filename.startswith(start):
               continue
            print(filename)
            if os.path.isdir(os.path.join(path1, filename)):
                path2 = f'{path1}/{filename}'
                i = 0
                routes = []
                results = defaultdict(int)
                results_crimes = []
                print(path2)
                for filename2 in os.listdir(path2):
                    if filename2.startswith('test'):
                        df = pd.read_csv(os.path.join(path2, filename2), sep=';')
                        for x in df['identificador'].unique():
                            results[x] += 1
                        d = defaultdict(int)
                        for n, x in df.groupby('patrol'):
                            route = []
                            for k, y in x.iterrows():
                                route.append(str(y['identificador']))
                                d[y['identificador']] += 1
                            routes.append(route)
                        i += 1
                        results_crimes.append(d)
                        if i == simulations:
                            break
                # print(i)
                r = calculate_pai(w, [3, 5, 10, 20], results, simulations)
                r2 = calculate_crimes(w, results_crimes)
                resultss[f'{patrols}_{filename}'] = r
                resultss2[f'{patrols}_{filename}'] = r2
                routes_model[f'{patrols}_{filename}'] = routes
    print('COVERAGE % of hotspots given random initial position')
    print('CONFIGURATION\t\t\t | 3%\t\t | 5%\t\t | 10%\t\t | 20%')
    print('-------------------------------------------------------------------------')
    # for x, y in conf.items():
    #     k = resultss[x]
    #     print(f'{y} \t\t& {k[3]:.3f} \t& {k[5]:.3f} \t& {k[10]:.3f} \t& {k[20]:.3f}')
    for x, y in sorted(resultss.items(), key=lambda x:x[0]):
        print(f'{x} \t\t& {y[3]} & {y[5]} & {y[10]} & {y[20]} & '
              f'{round(calculate_entropy(w, int(x.split("_")[0]), routes_model[x]), 2)} \\\\ %& {round(resultss2[x], 2)} \\\\')
    # for x in [3, 5, 10, 20]:
    #     print(x, end=' & ')
    #     print(" & ".join([f'{resultss[y][x] * 100:.1f}' for y in resultss]), end=" ")
    #     print("\\\\")
    # k = resultss[x]
    # print(f'{y} & {k[3] * 100:.1f} & {k[5] * 100:.1f} & {k[10] * 100:.1f} & {k[20] * 100:.1f} \\\\')


if __name__ == '__main__':
    main()
