import math
import random
import time
from copy import deepcopy

import numpy as np
from general_utils_j.umaths import map_value
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box, Discrete

from hotspots_simulation.agents import PolicePatrol, AbstractAgent
from hotspots_simulation.environment.environment import EnvironmentStreets
from hotspots_simulation.render_hotspots_patrols_routing import RenderHotpotsPatrolRouting, close
from utils.db_init import generate_dbs

policy_mapping_dict = {
    "hotspots_patrols_routing": {
        "description": "Patrols",
        "team_prefix": ("patrol_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
}


class HotspotsPatrolRouting(MultiAgentEnv):

    def __init__(self, env_config):
        # Get DB
        generate_dbs()

        # Obtener configuraciones del entorno
        self.config = deepcopy(env_config)
        self.metadata = {'name': "hotspots_patrols_routing", 'render.modes': ['human_mov', 'human_off']}
        self.max_steps = self.config.pop('max_steps', 50)
        self.area_algoritmo = self.config.pop('area_algortimo', 3)
        self.size_obs = self.config.pop('size_square_obs', 5)
        self.out_of_zone = self.config.pop('out_of_zone', -100)
        self.reward_exploration = self.config.pop('reward_exploration', 5)
        self.normalizer_crimes = self.config.pop('normalizer_crimes', 10)
        self.initial_position = self.config.pop('initial_position', 'random')

        self.original_number_patrols = self.config.pop('patrols', 5)
        self.original_number_agents = self.original_number_patrols

        self.number_patrols = self.original_number_patrols
        self.num_agents = self.number_patrols
        self.no_render = True
        self._seed = None

        self.id_actual_patrols = self.number_patrols + 1

        self.max_agents_alive = self.num_agents
        self.min_alive_patrols = self.original_number_patrols
        self.max_alive_patrols = self.original_number_patrols

        # Creamos el entorno y a sus agentes
        self.world = EnvironmentStreets(area=self.area_algoritmo, steps=self.max_steps, initial_position=self.initial_position)
        self._patrols = []
        for x in range(self.number_patrols):
            p = PolicePatrol(agent_id=x, memory_cells=self.max_steps)
            self.world.agents.append(p)
            self._patrols.append(p)
        self._agents = self.world.agents
        self.agents = [a.agent_id for a in self._agents]
        self.positions = [0] * self.number_patrols
        self.score = 0

        # Configuraciones para el render
        self.mode = self.config.pop('render', 'human_mov')
        self.nodes_color = None
        self.render_world = None

        # Definimos el espacio de observaciones y el espacio de acciones
        self.action_space = Discrete(9)

        self.observation_space = GymDict(
            {
                "obs": Box(-2, math.inf, shape=(self.number_patrols
                                                + (pow((self.size_obs * 2 + 1), 2) * 2
                                                   if self.size_obs > -1 else
                                                   len(self.world.get_block_cell_full('visits'))*2)
                                                # + self.world.total_cells*2
                                                ,),
                           dtype=np.dtype("float64")),
                # "state": Box(0, 1, shape=(4 * self.max_agents_alive,), dtype=np.dtype("float64")),
                "action_mask": Box(
                    0, 1, shape=(9,), dtype=np.dtype("float64")
                ),
            }
        )
        # self._obs_empty = np.zeros(shape=(3,), dtype=np.dtype("float64"))

        # Otros parametros
        self.steps = 0
        self.seed_env = None

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_steps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

    def seed(self, seed=None):
        if self._seed == None:
            random.seed(seed)
            np.random.seed(seed)
            self.world.seed(seed)
            self.seed_env = seed
        else:
            random.seed(self._seed)
            np.random.seed(self._seed)
            self.world.seed(self._seed)
            self.seed_env = self._seed


    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        self.steps = 0
        self.score = 0
        self.positions = [0] * self.number_patrols
        self.number_patrols = self.original_number_patrols
        self.num_agents = self.number_patrols
        self.world.reset()
        for i, a in enumerate(self._patrols):
            a.set_agent_id(i)
            a.score = 0
            self.positions[a.numeric_id] = a.cell.identificador
        self.agents = [a.agent_id for a in self._agents]
        self.id_actual_citizens = self.number_patrols + 1
        # Reinicio del mundo
        return self._get_all_obs_action_mask_dict()

    def step(self, action_dict: dict):
        #
        # Movimiento
        #

        self.positions = [0] * self.number_patrols
        for a in self._patrols:
            self.world.move_to(a, action_dict[a.agent_id])
            self.positions[a.numeric_id] = a.cell.identificador

        self.steps += 1
        dones = {"__all__": self.steps >= self.max_steps}
        infos = {i: {} for i in self.agents}
        individual_rewards = [self._get_reward(i) for i in self._agents]
        rews = {i.agent_id: sum(individual_rewards) + individual_rewards[e] for e, i in enumerate(self._agents)}
        return_obs = self._get_all_obs_action_mask_dict()
        return return_obs, rews, dones, infos

    def _get_obs(self, agent, world_crimes, visitas):
        a: AbstractAgent = agent
        return np.array([self.positions[a.numeric_id]]
                        + self.positions[:a.numeric_id]
                        + self.positions[a.numeric_id + 1:]
                        + world_crimes + visitas,  # Crimenes estimados de cada celda [num_cell]
                        # + areas,  # 0 Si la celda no pertenece al area y 1 si si [num_cell]
                        dtype=np.dtype("float64"))

    def _get_obs_window(self, a):
        return np.concatenate((np.array([self.positions[a.numeric_id]]
                                        + self.positions[:a.numeric_id] + self.positions[a.numeric_id + 1:],
                                        dtype=np.dtype("float64")),
                               self.world.get_block_cells(a.cell, l_sight=self.size_obs, stat='crimes'),
                               self.world.get_block_cells(a.cell, l_sight=self.size_obs, stat='visits')))

    def _get_reward(self, agent: PolicePatrol):
        if agent.cell.zona_algoritmo != str(self.area_algoritmo):
            return self.out_of_zone
        extra = 0
        if agent.cell.visitas_patrols == 1 and agent.cell.estimated_crimes > 0:
            extra = self.reward_exploration
            if agent.cell.estimated_crimes >= 10:
                extra = extra * 10
        if (agent.cell.estimated_crimes / agent.cell.visitas_patrols) / self.normalizer_crimes < 1:
            extra += self.out_of_zone / 2
        return (agent.cell.estimated_crimes / agent.cell.visitas_patrols) / self.normalizer_crimes + extra

    def _get_reward_comb(self, agent: PolicePatrol):
        if agent.cell.zona_algoritmo != str(self.area_algoritmo):
            return self.out_of_zone
        extra = 0
        d = ((agent.cell.estimated_crimes / agent.cell.visitas_patrols) / self.normalizer_crimes)
        if agent.cell.visitas_patrols == 1 and d > 1:
            extra = self.reward_exploration
            if agent.cell.estimated_crimes >= 10:
                extra = extra * 10
        elif d <= 1:
            extra += self.out_of_zone / 2
        return d + extra

    def _get_reward_score(self, agent: PolicePatrol):
        min_reward = -1
        max_reward = 10
        if agent.cell.zona_algoritmo != str(self.area_algoritmo):
            agent.score += min_reward
        else:
            agent.score += agent.cell.estimated_crimes / agent.cell.visitas_patrols
        return map_value(agent.score, 0,
                         self.world.total_estimated_crimes, min_reward, max_reward,
                         False, min_reward, max_reward)

    # def _get_state(self, obs):
    #     if obs is None:
    #         obs = self._get_all_obs()
    #     state = np.concatenate(obs, axis=0)
    #     increment = self.max_agents_alive - self.num_agents
    #     if increment > 0:
    #         state = np.concatenate([state] + ([self._obs_empty] * increment))
    #     return state

    @staticmethod
    def _get_mask(agent: AbstractAgent):
        return agent.cell.vecinos_mask

    def _get_all_obs(self):
        # crimes_estimated = self.world.get_cells_estimated_cells()
        # visitas = self.world.get_cells_visited()
        # area = self.world.areas_obs
        return [
            # self._get_obs(a, crimes_estimated, visitas)
            self._get_obs_window(a)
            for a in self._agents]

    def _get_all_obs_action_mask_list(self):
        obs = self._get_all_obs()
        # state = self._get_state(obs)
        return [{'obs': o,
                 # 'state': state,
                 'action_mask': self._get_mask(a)} for a, o in zip(self._agents, obs)]

    def _get_all_obs_action_mask_dict(self):
        obs = self._get_all_obs()
        # state = self._get_state(obs)
        return {a.agent_id: {'obs': o,
                             # 'state': state,
                             'action_mask': self._get_mask(a)} for a, o in zip(self._agents, obs)}

    # RENDER
    def render(self, mode=None):
        if self.mode == 'human_mov' or self.mode == 'human_off':
            if self.render_world is None:
                self.render_world = RenderHotpotsPatrolRouting(self.world, render=self.no_render)
            r = sum(self._get_reward(a) for a in self._agents) * len(self._agents)
            self.score += r
            self.render_world.render(self.mode, self.steps, self.max_steps, self.score, r)
            time.sleep(0.5)
            return True
        return False

    def get_paths(self):
        return self.world.get_paths()

    def close(self):
        """Close the environment"""
        if self.render_world is not None:
            close()
