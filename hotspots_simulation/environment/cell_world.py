from collections import defaultdict
from dataclasses import dataclass
from typing import List

from numpy import ndarray


@dataclass(frozen=True)
class CellWorld:
    v5: tuple
    identificador: int
    zona_algoritmo: str

    position: tuple[int, int]

    crime_rate: float
    estimated_crimes: int
    real_crimes: float

    vecinos_v5: list
    vecinos_identificadores: list

    vecinos_actions: ndarray
    vecinos_mask: ndarray


class CellWorldSimulation:

    def __init__(self, cell: CellWorld):
        self.__cell = cell
        self.visitas_patrols = 0
        self.visitas_agentes = defaultdict(int)
        self.current_patrols = set()
        self.current_citizens = set()
        self.current_offenders = set()

    @property
    def v5(self) -> tuple:
        return self.__cell.v5

    @property
    def identificador(self) -> int:
        return self.__cell.identificador

    @property
    def position(self) -> tuple[int, int]:
        return self.__cell.position

    @property
    def crime_rate(self) -> float:
        return self.__cell.crime_rate

    @property
    def real_crimes(self) -> float:
        return self.__cell.real_crimes

    @property
    def estimated_crimes(self) -> int:
        return self.__cell.estimated_crimes

    @property
    def zona_algoritmo(self) -> str:
        return self.__cell.zona_algoritmo

    @property
    def vecinos_v5(self) -> List[tuple]:
        return self.__cell.vecinos_v5

    @property
    def vecinos_identificadores(self) -> List[int]:
        return self.__cell.vecinos_identificadores

    @property
    def vecinos_actions(self) -> ndarray:
        return self.__cell.vecinos_actions

    @property
    def vecinos_mask(self) -> ndarray:
        return self.__cell.vecinos_mask

    def visitas_agente(self, agent_id):
        return self.visitas_agentes[agent_id]

    def add_agent(self, agent_id, type_agent):
        ex = True
        if type_agent == 'offender':
            ex = self.add_offender(agent_id)
        elif type_agent == 'citizen':
            ex = self.add_citizen(agent_id)
        elif type_agent == 'patrol':
            ex = self.add_patrol(agent_id)
        if not ex:
            self.visitas_agentes[agent_id] += 1

    def remove_agent(self, agent_id, type_agent):
        if type_agent == 'offender':
            self.remove_offender(agent_id)
        elif type_agent == 'citizen':
            self.remove_citizen(agent_id)
        elif type_agent == 'patrol':
            self.remove_patrol(agent_id)

    def add_offender(self, agent_id):
        ex = agent_id in self.current_offenders
        self.current_offenders.add(agent_id)
        return ex

    def remove_offender(self, agent_id):
        self.current_offenders.discard(agent_id)

    def add_citizen(self, agent_id):
        ex = agent_id in self.current_citizens
        self.current_citizens.add(agent_id)
        return ex

    def remove_citizen(self, agent_id):
        self.current_citizens.discard(agent_id)

    def add_patrol(self, agent_id):
        ex = agent_id in self.current_patrols
        self.current_patrols.add(agent_id)
        self.visitas_patrols += 1
        return ex

    def remove_patrol(self, agent_id):
        self.current_patrols.discard(agent_id)

    @property
    def current_agents_cell(self):
        return self.current_offenders_cell + self.current_citizens_cell + self.current_patrols_cell

    @property
    def current_offenders_cell(self):
        return len(self.current_offenders)

    @property
    def current_citizens_cell(self):
        return len(self.current_citizens)

    @property
    def current_patrols_cell(self):
        return len(self.current_patrols)

    def reset(self):
        self.visitas_agentes = defaultdict(int)
        self.visitas_patrols = 0
        self.current_offenders = set()
        self.current_citizens = set()
        self.current_patrols = set()

    def __repr__(self):
        return f'{self.identificador} -> {self.estimated_crimes}'
