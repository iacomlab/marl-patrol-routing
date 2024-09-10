import os
from collections import defaultdict

import numpy as np
import yaml

from hotspots_simulation.environment.environment import EnvironmentStreets
from hotspots_simulation.wrapper_hotspots_patrol_routing import HotspotsPatrolRouting
from main_refinate_results import calculate_pai, calculate_crimes, calculate_entropy


def get_config_env(name):
    env_config_file_path = os.path.join(os.path.dirname(__file__),
                                        f"configurations//{name}.yaml")
    with open(env_config_file_path, "r") as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    return env_config_dict


def get_actions(obs, line_sight=6):
    actions = {}
    total_patrols = len(obs)
    prev = line_sight - 1
    post = line_sight + 1
    for patrol_id, obs_patrol in obs.items():
        real_obs = obs_patrol['obs'][total_patrols:]
        actions_mask = obs_patrol['action_mask']
        crimes, vis = np.split(real_obs, 2)
        crimes = np.split(crimes, line_sight * 2 + 1)
        posibilidades = [crimes[prev][prev] if actions_mask[0] == 1 else -1,
                         crimes[prev][line_sight] if actions_mask[1] == 1 else -1,
                         crimes[prev][post] if actions_mask[2] == 1 else -1,
                         crimes[line_sight][prev] if actions_mask[3] == 1 else -1,
                         crimes[line_sight][line_sight] if actions_mask[4] == 1 else -1,
                         crimes[line_sight][post] if actions_mask[5] == 1 else -1,
                         crimes[post][prev] if actions_mask[6] == 1 else -1,
                         crimes[post][line_sight] if actions_mask[7] == 1 else -1,
                         crimes[post][post] if actions_mask[8] == 1 else -1]
        chosen = max(enumerate(posibilidades), key=lambda x: x[1])[0]
        actions[patrol_id] = chosen
    return actions


def main():
    area = 3
    w = EnvironmentStreets(area, 50, clean_not_accessible=True)
    env_dict_2 = get_config_env('hotspots_patrols_routing')
    results_f = {}
    resultss_f = {}
    routes_d = {}
    for initial in ['random', 'best']:
        # initial = 'best'
        for kk in [2, 5, 10]:
            iden = f'{initial}_{kk}'
            not_default_args = {'patrols': kk, 'size_square_obs': 6, 'area_algortimo': area, 'max_steps': 50,
                                'initial_position': initial}
            env_dict = {**env_dict_2["env_args"], **not_default_args}
            hotspot_patrol_routing = HotspotsPatrolRouting(env_dict)
            results = defaultdict(int)
            resultsss = []
            ran = range(0, 100) if initial == 'random' else [0]
            routes = []
            for i in ran:
                print(i)
                obs = hotspot_patrol_routing.reset()
                routes_agent = {}
                for x in hotspot_patrol_routing.world.agents:
                    routes_agent[x.agent_id] = [str(x.cell.identificador)]

                # hotspot_patrol_routing.render('human_off')
                # time.sleep(5)
                obs, reward, done, info = hotspot_patrol_routing.step(get_actions(obs, 6))
                for x in hotspot_patrol_routing.world.agents:
                    routes_agent[x.agent_id].append(str(x.cell.identificador))
                # hotspot_patrol_routing.render('human_off')
                while not done['__all__']:
                    obs, reward, done, info = hotspot_patrol_routing.step(get_actions(obs, 6))
                    for x in hotspot_patrol_routing.world.agents:
                        routes_agent[x.agent_id].append(str(x.cell.identificador))
                d = {}
                for x in filter(lambda h: h.visitas_patrols > 0, hotspot_patrol_routing.world.cells.values()):
                    d[x.identificador] = x.visitas_patrols
                    results[x.identificador] += 1
                resultsss.append(d)
                for x in routes_agent.values():
                    routes.append(x)
            r = calculate_pai(w, [3, 5, 10, 20], results, len(ran))
            r2 = calculate_crimes(w, resultsss)
            results_f[iden] = r
            resultss_f[iden] = r2
            routes_d[iden] = routes
    l = [3, 5, 10, 20]
    for x, y in results_f.items():
        print(x)
        for z in l:
            print(f'{y[z]:.3f}', end=' & ')
        print(f'{calculate_entropy(w, int(x.split("_")[1]), list(routes_d[x]), 50):.2f} & ', end='')
        print(f'{resultss_f[x]:.2f} \\\\')
        print()


if __name__ == '__main__':
    main()
