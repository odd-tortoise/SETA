# agents/snode.py

from .base import Agent

class ChildNode(Agent):
    """
    Child-node: (age, length, rank, temp), evolves via logistic ODE.
    """
    def __init__(self, state):
        # state must have .age, .length, .rank, .temperature
        self.state = state

    def sense(self, env):
        self.state.temperature = env.get_temperature()

    def evolve(self, dt, solver=None):
        # If solver is provided, it can handle a batch; here we do scalar Euler:
        Lmax = self.state.rank * (1 + self.state.temperature / 30.0)
        r = 0.5
        dL = r * self.state.length * (1 - self.state.length / Lmax)
        self.state.length += dL * dt
        self.state.age += dt

    def think(self, mode='inference'):
        # No decision logic for S-nodes
        pass

    def act(self):
        # S-nodes have no actions
        pass
