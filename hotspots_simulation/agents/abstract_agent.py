from abc import ABC, abstractmethod

import numpy as np

from hotspots_simulation.environment.cell_world import CellWorldSimulation


class MemoryCell:
    __slots__ = ['memory_cells']

    def __init__(self, shape):
        self.memory_cells = np.empty(shape, dtype=int)
        self.memory_cells[:] = -1

    def unique_elements_counter(self):
        unique_elements, counter = np.unique(self.memory_cells, return_counts=True)
        if unique_elements[0] == -1:
            unique_elements = unique_elements[1:]
            counter = counter[1:]
        return dict(zip(unique_elements, counter))

    def counter_element(self, identificador):
        return self.unique_elements_counter().get(identificador, 0)

    def add_cell(self, cell):
        self.memory_cells[:-1] = self.memory_cells[1:]
        self.memory_cells[-1] = int(cell)

    def reset(self):
        self.memory_cells[:] = -1

    def __repr__(self):
        return f'{self.memory_cells}'


class AgentException(Exception):
    ...


class AbstractAgent(ABC):

    def __init__(self, agent_id, memory_cells):
        self.__agent_id = None
        self.__numeric_id = None
        self.set_agent_id(agent_id)
        self.__cell = None
        self.__memory_cell = MemoryCell(memory_cells)
        self.steps_alive = -1

    def set_agent_id(self, agent_id):
        self.__agent_id = f'{self.type_agent()}_{agent_id}'
        self.__numeric_id = agent_id

    @property
    def agent_id(self):
        return self.__agent_id

    @property
    def numeric_id(self):
        return self.__numeric_id

    @property
    def name(self):
        return self.__agent_id

    @property
    def memory_cell(self):
        return self.__memory_cell

    def change_cell(self, cell: CellWorldSimulation):
        if self.__cell is not None:
            self.__cell.remove_agent(self.agent_id, self.type_agent())
        self.__cell = cell
        self.__cell.add_agent(self.agent_id, self.type_agent())
        self.__memory_cell.add_cell(cell.identificador)

    def set_init_cell(self, cell: CellWorldSimulation):
        self.__cell = cell
        self.__cell.add_agent(self.agent_id, self.type_agent())

    def delete_agent(self):
        self.__cell.remove_agent(self.agent_id, self.type_agent())
        del self.__memory_cell

    @property
    def cell(self) -> CellWorldSimulation:
        return self.__cell

    def __str__(self):
        return f'AbstractAgent {self.agent_id}'

    def __repr__(self):
        return f'AbstractAgent {self.agent_id}'

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.agent_id)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.agent_id == other.agent_id

    def __lt__(self, other):
        if not issubclass(type(other), AbstractAgent):
            raise AgentException(f'{other} -> No es un agente')
        if isinstance(other, self.__class__):
            return self.agent_id < other.agent_id
        else:
            return str(self.__class__.__name__) < str(other.__class__.__name__)

    def __gt__(self, other):
        if not issubclass(type(other), AbstractAgent):
            raise AgentException(f'{other} -> No es un agente')
        if isinstance(other, self.__class__):
            return self.agent_id > other.agent_id
        else:
            return str(self.__class__.__name__) > str(other.__class__.__name__)

    @staticmethod
    @abstractmethod
    def type_agent() -> str:
        ...

    @staticmethod
    @abstractmethod
    def first_type() -> str:
        ...

    @staticmethod
    @abstractmethod
    def priority() -> int:
        ...

    @staticmethod
    @abstractmethod
    def color() -> str:
        ...

    def reset_memory(self):
        self.memory_cell.reset()
