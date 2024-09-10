from hotspots_simulation.agents import AbstractAgent


class PolicePatrol(AbstractAgent):

    def __init__(self, agent_id, memory_cells):
        super().__init__(agent_id, memory_cells)
        self.score = 0

    def __str__(self):
        return f'PolicePatrol {self.agent_id}'

    def __repr__(self):
        return f'PolicePatrol {self.agent_id}'

    @staticmethod
    def first_type() -> str:
        return 'patrol'

    @staticmethod
    def type_agent() -> str:
        return 'patrol'

    @staticmethod
    def priority() -> int:
        return 2

    @staticmethod
    def color() -> str:
        return 'blue'

