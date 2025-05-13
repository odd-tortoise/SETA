# agents/anode.py

from .base import Agent

class GenNode(Agent):
    """
    Gen-node: keeps track of (age, temp) and will inject a thinker later.
    """
    def __init__(self, state, thinker=None):
        # state is a simple object with .age and .temperature attributes
        self.state = state
        self.thinker = thinker
        self._pending_action = None

    def sense(self, env):
        # Synchronize temperature
        self.state.temperature = env.get_temperature()

    def evolve(self, dt, solver=None):
        # Age increases linearly
        self.state.age += dt

    def think(self, mode='inference'):
        if self.thinker is None:
            self._pending_action = False
        else:
            features = [self.state.age, self.state.temperature]
            self._pending_action = self.thinker.act(features, mode)

    def act(self):
        # Return the decision to spawn (True/False)
        return self._pending_action
