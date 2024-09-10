from typing import TypeVar
from .abstract_agent import AbstractAgent

from .police import PolicePatrol

AgenteI = TypeVar('AgenteI', bound=AbstractAgent)

__all__ = [
    'AbstractAgent',
    'PolicePatrol',
    'AgenteI'
]
