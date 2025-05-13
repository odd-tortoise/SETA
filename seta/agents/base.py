# agents/base.py

from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Base class for all agents/nodes in SETA.
    Defines the sense, evolve, think, act interface.
    """
    @abstractmethod
    def sense(self, env):
        pass

    @abstractmethod
    def evolve(self, dt, solver=None):
        pass

    @abstractmethod
    def think(self, mode='inference'):
        pass

    @abstractmethod
    def act(self):
        pass

    