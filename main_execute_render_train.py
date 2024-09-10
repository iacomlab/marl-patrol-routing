from dotenv import load_dotenv
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

from hotspots_simulation.wrapper_hotspots_patrol_routing import HotspotsPatrolRouting


def trial_dirname_creator_func(t):
    a, c, *_ = t.custom_trial_name.split('-')
    from datetime import datetime
    b = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    return f'{a}_{b}_{c}'


def trial_name_creator_func(t):
    return f'{t._trainable_name(False)}-{t.trial_id}'


ENV_REGISTRY["hotspots_patrols_routing"] = HotspotsPatrolRouting
COOP_ENV_REGISTRY["hotspots_patrols_routing"] = HotspotsPatrolRouting


# {algorithm}_gru_hotspots_patrols_routing/
def render_env(algorithm, name, checkpoint, patrols):
    direccion = f'good_position/{name}'
    checkpoint_2 = '0' * (6 - len(checkpoint)) + checkpoint

    env = marl.make_env(environment_name="hotspots_patrols_routing", map_name='hotspots_patrols_routing',
                        abs_path="configurations/hotspots_patrols_routing.yaml",
                        log_init=False, force_coop=True, **{'patrols': patrols})
    # pick mappo algorithms
    c_alg_None = f'configurations/algorithms/common/{algorithm}_server_cm.yaml'
    algo = marl.algos.__getattribute__(algorithm)(
        hyperparam_source=c_alg_None,
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


def main():
    alg = 'vdppo'
    save = 'VDPPOTrainer_hotspots_6.0_6.0--2024-05-21_23-00'
    checkpoint = '3334'
    render_env(alg, save, checkpoint, 6)


if __name__ == '__main__':
    load_dotenv()
    main()
