import argparse
import csv
import time

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

from dotenv import load_dotenv

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

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c_env", "--conf_env",
                           help="direction to yaml of your env configuration",
                           default='configurations/hotspots_patrols_routing.yaml')

    argParser.add_argument("-ie", "--identificator_env",
                           help="identificator of the params of the environment present in the csv",
                           default=1)

    argParser.add_argument("-a", "--algorithm",
                           help="algorithm to execute",
                           default='mappo')

    argParser.add_argument("-c_alg", "--conf_algorithm",
                           help="direction to yaml of your algorithm configuration",
                           default=None)

    args = argParser.parse_args()
    load_dotenv()
    env_config_dict = {}
    with open("env_params_server_train_revista_2.csv", "r", encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=';')
        rows = [row for row in reader]
        env_config_dict = {x: (float(y) if x != 'env.initial_position' else y)
                           for x, y in zip(rows[0][1:], rows[int(args.identificator_env)][1:])}
    algorithm = {x[4:]: y for x, y in env_config_dict.items() if x.startswith('alg')}
    env_config_dict_2 = {x[4:]: (int(y) if x != 'env.initial_position' else y) for x, y in env_config_dict.items() if x.startswith('env')}
    ENV_REGISTRY["hotspots_patrols_routing"] = HotspotsPatrolRouting
    COOP_ENV_REGISTRY["hotspots_patrols_routing"] = HotspotsPatrolRouting
    env = marl.make_env(environment_name="hotspots_patrols_routing", map_name='hotspots_patrols_routing',
                        abs_path=args.conf_env, log_init=False, force_coop=True, **env_config_dict_2)
    # pick mappo algorithms
    c_alg_None = f'configurations/algorithms/common/{args.algorithm}_server_cm.yaml'
    print(args.algorithm)
    alg = marl.algos.__getattribute__(args.algorithm)(
        hyperparam_source=args.conf_algorithm if args.conf_algorithm is not None else c_alg_None,
        absolute_path_hyper=True, **algorithm)

    def trial_dirname_creator_func(t):
        from datetime import datetime
        b = datetime.today().strftime("%Y-%m-%d_%H-%M")
        return f'{t}--{b}'

    # noinspection PyProtectedMember
    def trial_name_creator_func(t):
        return f'{"_".join(t._trainable_name(False).split("_")[:2])}_{"_".join(str(x) for x in env_config_dict.values())}'
    # customize model
    model = marl.build_model(env, alg, {"core_arch": "gru", "encode_layer": "256-256"}
                                        # "encode_layer": "128-128"}
                             )
    alg.fit(env, model, stop={"training_iteration": 10_000, "timesteps_total": 100_000_000}, num_gpus=1,
            num_workers=20, local_mode=False, checkpoint_freq=250,
            share_policy=diccionario_params[args.algorithm]['share_policy'],
            trial_name_creator=trial_name_creator_func, trial_dirname_creator=trial_dirname_creator_func)
    time.sleep(10)
