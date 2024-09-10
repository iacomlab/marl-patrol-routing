import os
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from hotspots_simulation.wrapper_hotspots_patrol_routing import HotspotsPatrolRouting


def train_server(trial_name_creator_func, trial_dirname_creator_func):
    env = marl.make_env(environment_name="hotspots_patrols_routing", map_name='hotspots_patrols_routing',
                        abs_path="configurations/hotspots_patrols_routing.yaml", log_init=False)
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="configurations/mappo_server_test.yaml", absolute_path_hyper=True)
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "gru", "encode_layer": "128-128"})
    mappo.fit(env, model, stop={'training_iteration': 200}, num_gpus=1,
              num_workers=0, local_mode=True, checkpoint_freq=50, share_policy='group',
              trial_name_creator=trial_name_creator_func, trial_dirname_creator=trial_dirname_creator_func)

def train_server_checkpoint(trial_name_creator_func, trial_dirname_creator_func):
    alg = 'ippo'
    save = 'IPPOTrainer_hotspots_--2024-03-04_14-00'
    direccion = f'{alg}_gru_hotspots_patrols_routing/{save}'
    checkpoint = '25000'
    checkpoint_2 = '0' * (6 - len(checkpoint)) + checkpoint
    env = marl.make_env(environment_name="hotspots_patrols_routing", map_name='hotspots_patrols_routing',
                        abs_path="configurations/hotspots_patrols_routing.yaml")
    # pick mappo algorithms
    algo = marl.algos.__getattribute__(alg)(
        hyperparam_source=f"configurations/algorithms/common/{alg}_server_cm.yaml",
        absolute_path_hyper=True)
    # customize model
    model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "256-128"})
    algo.fit(env, model,
             restore_path={'params_path': f"exp_results/{direccion}/params.json",
                           # experiment configuration
                           'model_path': f"exp_results/{direccion}/"
                                         f"checkpoint_{checkpoint_2}/checkpoint-{checkpoint}",
                           # checkpoint path
                           },  # render
             local_mode=False,
             num_workers=2,
             render_env=False,
             num_gpus=0, num_gpus_per_worker=0,
             share_policy="group",
             trial_name_creator=trial_name_creator_func, trial_dirname_creator=trial_dirname_creator_func,
             checkpoint_end=False)

def render_env(trial_name_creator_func, trial_dirname_creator_func):
    alg = 'vdppo'
    save = 'test_1'
    direccion = f'{alg}_gru_hotspots_patrols_routing/{save}'
    checkpoint = '4000'
    checkpoint_2 = '0' * (6 - len(checkpoint)) + checkpoint
    # identificador = 18
    # with open("env_params_server_train.csv", "r", encoding='utf-8-sig') as f:
    #     import csv
    #     reader = csv.reader(f, delimiter=';')
    #     rows = [row for row in reader]
    #     env_config_dict = {x[4:]: int(y) for x, y in zip(rows[0][1:], rows[identificador][1:])}
    env = marl.make_env(environment_name="hotspots_patrols_routing", map_name='hotspots_patrols_routing',
                        abs_path="configurations/hotspots_patrols_routing.yaml")
    # pick mappo algorithms
    algo = marl.algos.__getattribute__(alg)(
        hyperparam_source=f"configurations/algorithms/common/{alg}_server_cm.yaml",
        absolute_path_hyper=True)
    # customize model
    model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "256-256"})
    algo.render(env, model,
                restore_path={'params_path': f"exp_results/{direccion}/params.json",
                              # experiment configuration
                              'model_path': f"exp_results/{direccion}/"
                                            f"checkpoint_{checkpoint_2}/checkpoint-{checkpoint}",
                              # checkpoint path
                              'render': True},  # render
                local_mode=True,
                num_workers=0,
                render_env=True,
                num_gpus=0, num_gpus_per_worker=0,
                share_policy="group",
                trial_name_creator=trial_name_creator_func, trial_dirname_creator=trial_dirname_creator_func,
                checkpoint_end=False)


def test_chaeckpoint(trial_name_creator_func, trial_dirname_creator_func):
    print('Test')
    alg = 'vdppo'
    save = 'test_1'
    direccion = f'{alg}_gru_hotspots_patrols_routing/{save}'
    checkpoint = '4000'
    checkpoint_2 = '0' * (6 - len(checkpoint)) + checkpoint
    # identificador = 18
    # with open("env_params_server_train.csv", "r", encoding='utf-8-sig') as f:
    #     import csv
    #     reader = csv.reader(f, delimiter=';')
    #     rows = [row for row in reader]
    #     env_config_dict = {x[4:]: int(y) for x, y in zip(rows[0][1:], rows[identificador][1:])}
    env = marl.make_env(environment_name="hotspots_patrols_routing", map_name='hotspots_patrols_routing',
                        abs_path="configurations/hotspots_patrols_routing.yaml")
    # pick mappo algorithms
    algo = marl.algos.__getattribute__(alg)(
        hyperparam_source=f"configurations/algorithms/common/{alg}_server_cm.yaml",
        absolute_path_hyper=True)
    # customize model
    model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "256-256"})
    algo.fit(env, model,
             restore_path={'params_path': f"exp_results/{direccion}/params.json",
                           # experiment configuration
                           'model_path': f"exp_results/{direccion}/"
                                         f"checkpoint_{checkpoint_2}/checkpoint-{checkpoint}",
                           # checkpoint path
                           },  # render
             local_mode=False,
             num_workers=1,
             render_env=False,
             num_gpus=0, num_gpus_per_worker=0,
             share_policy="group",
             trial_name_creator=trial_name_creator_func, trial_dirname_creator=trial_dirname_creator_func,
             checkpoint_end=False)


def main():
    def trial_dirname_creator_func(t):
        a, c, *_ = t.custom_trial_name.split('-')
        from datetime import datetime
        b = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        return f'{a}_{b}_{c}'

    # noinspection PyProtectedMember
    def trial_name_creator_func(t):
        return f'{t._trainable_name(False)}-{t.trial_id}'

    ENV_REGISTRY["hotspots_patrols_routing"] = HotspotsPatrolRouting
    COOP_ENV_REGISTRY["hotspots_patrols_routing"] = HotspotsPatrolRouting
    # register new env
    if os.getenv('MACHINE', 'local') == 'local':
        render = True
        if not render:
            test_chaeckpoint(trial_name_creator_func, trial_dirname_creator_func)
            # env = marl.make_env(environment_name="hotspots_patrols_routing", map_name='hotspots_patrols_routing',
            #                     abs_path="configurations/hotspots_patrols_routing.yaml")
            # # pick mappo algorithms
            # mappo = marl.algos.mappo(hyperparam_source="configurations/mappo_local.yaml", absolute_path_hyper=True)
            # # customize model
            # model = marl.build_model(env, mappo, {"core_arch": "gru", "encode_layer": "128-128"})
            # mappo.fit(env, model, stop={'timesteps_total': 100_000},
            #           local_mode=True,
            #           num_workers=0,
            #           num_gpus=0, num_gpus_per_worker=0, checkpoint_freq=10, share_policy='group',
            #           trial_name_creator=trial_name_creator_func, trial_dirname_creator=trial_dirname_creator_func)
        else:
            render_env(trial_name_creator_func, trial_dirname_creator_func)
    else:
        test = True
        if test == True:
            test_chaeckpoint(trial_name_creator_func, trial_dirname_creator_func)
        else:
            train_server(trial_name_creator_func, trial_dirname_creator_func)


if __name__ == '__main__':
    main()
