import argparse
import csv
import os
import time

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

from hotspots_simulation.wrapper_hotspots_patrol_routing import HotspotsPatrolRouting

diccionario_params = {
    'qmix': {'share_policy': 'all'},
    'iql': {'share_policy': 'all'},
    'vdn': {'share_policy': 'all'},
    'ia2c': {'share_policy': 'group'},
    'maa2c': {'share_policy': 'group'},
    'coma': {'share_policy': 'group'},
    'vda2c': {'share_policy': 'group'},
    'itrpo': {'share_policy': 'group'},
    'matrpo': {'share_policy': 'group'},
    'hatrpo': {'share_policy': 'group'},
    'ippo': {'share_policy': 'group'},
    'mappo': {'share_policy': 'group'},
    'maddpg': {'share_policy': 'group'},
    'vdppo': {'share_policy': 'group'},
    'happo': {'share_policy': 'group'},
}

nombres = {
    'vdppo': 'VDPPOTrainer_hotspots_',
    'ippo': 'IPPOTrainer_hotspots_',
    'mappo': 'MAPPOTrainer_hotspots_'
}


def trial_dirname_creator_func(t):
    from datetime import datetime
    b = datetime.today().strftime("%Y-%m-%d_%H-%M")
    return f'{t}--{b}'


# noinspection PyProtectedMember
def trial_name_creator_func(t):
    return f'{"_".join(t._trainable_name(False).split("_")[:2])}_{"_".join(str(x) for x in env_config_dict.values())}'


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-ie", "--identificator_env",
                           help="identificator of the params of the environment present in the csv",
                           default=2)
    argParser.add_argument("-a", "--algorithm",
                           help="algorithm to execute",
                           default='vdppo')
    args = argParser.parse_args()

    with open("env_params_server_simulate_revista.csv", "r", encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        rows = [row for row in reader]
        env_config_dict = {x:  (float(y) if x != 'env.initial_position' else y) for x, y in zip(rows[0][1:], rows[int(args.identificator_env)][1:])}
    algorithm = {x[4:]: y for x, y in env_config_dict.items() if x.startswith('alg')}
    env_config_dict_2 = {x[4:]: (int(y) if x != 'env.initial_position' else y) for x, y in env_config_dict.items() if x.startswith('env')}
    ENV_REGISTRY["hotspots_patrols_routing"] = HotspotsPatrolRouting
    COOP_ENV_REGISTRY["hotspots_patrols_routing"] = HotspotsPatrolRouting
    # n = f'{nombres[args.algorithm]}6.0_{env_config_dict_2["patrols"]}.0_{env_config_dict_2["out_of_zone"]}.0_{env_config_dict_2["normalizer_crimes"]}.0_{env_config_dict_2["reward_exploration"]}.0_{env_config_dict_2["max_steps"]}.0--'
    n = (f'{nombres[args.algorithm]}{env_config_dict_2["area_algortimo"]}.0_{env_config_dict_2["patrols"]}.0_{env_config_dict_2["size_square_obs"]}.'
         f'0_{env_config_dict_2["initial_position"]}--')
    if str(env_config_dict_2["area_algortimo"]) == '10':
        n = (f'{nombres[args.algorithm]}{env_config_dict_2["area_algortimo"]}.0_{env_config_dict_2["patrols"]}.0_{env_config_dict_2["size_square_obs"]}.'
         f'0_{env_config_dict_2["initial_position"]}_{env_config_dict_2["out_of_zone"]}.0_{env_config_dict_2["reward_exploration"]}.0--')
    # {args.algorithm}
    # _gru_hotspots_patrols_routing_2 /
    direccion = f'exp_results/simulates/'
    nombre = None
    for filename in os.listdir(direccion):
        if filename.startswith(n):
            nombre = filename
            break
    if nombre is None:
        print(n)
        raise Exception(1)
    direccion = f'{direccion}{nombre}'
    env = marl.make_env(environment_name="hotspots_patrols_routing", map_name='hotspots_patrols_routing',
                        abs_path="configurations/hotspots_patrols_routing.yaml",
                        log_init=False, force_coop=True,  **env_config_dict_2)
    alg = marl.algos.__getattribute__(args.algorithm)(
        hyperparam_source=f'configurations/algorithms/common/{args.algorithm}_server_cm.yaml',
        absolute_path_hyper=True)

    all_subdirs = sorted([d for d in os.listdir(direccion) if os.path.isdir(os.path.join(direccion, d))], key=lambda x:x)
    check = all_subdirs[-1]
    check_2 = check.split('_')[1]
    check_2 = check_2.lstrip('0')
    model = marl.build_model(env, alg, {"core_arch": "gru", "encode_layer": "256-256"}
                             )
    alg.render(env, model,
               restore_path={'params_path': f"{direccion}/params.json",
                             # experiment configuration
                             'model_path': f"{direccion}/"
                                           f"{all_subdirs[-1]}/checkpoint-{check_2}",
                             # checkpoint path
                             'render': True},  # render
               local_mode=True,
               num_workers=0,
               render_env=True,
               num_gpus=0,
               num_gpus_per_worker=0,
               share_policy="group",
               trial_name_creator=trial_name_creator_func, trial_dirname_creator=trial_dirname_creator_func,
               checkpoint_end=False)
    time.sleep(10)
