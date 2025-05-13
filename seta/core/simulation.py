# core.py

from typing import List
import numpy as np

class Simulation:
    """
    Drives the SETA loop over a list of agents.
    Assumes the first agent is an ANode, the rest are SNodes.
    """
    def __init__(self, environment, agents: List, dae_solver=None):
        self.env = environment
        self.agents = agents
        self.dae = dae_solver

    def step(self, t: float, dt: float):
        # 1) Sense
        for ag in self.agents:
            ag.sense(self.env)

        # 2) Evolve
        #   - For SNodes you might batch via dae_solver, but here we loop
        for ag in self.agents:
            ag.evolve(dt, solver=self.dae)

        # 3) Think
        for ag in self.agents:
            ag.think(mode='inference')

        # 4) Act
        new_agents = []
        for ag in self.agents:
            action = ag.act()
            # If an ANode decides to spawn, we expect it to return True
            if action:
                # create a fresh SNode; user must define initial state
                new_s = type(self.agents[1].state)()  # naive copy of state type
                from agents.child import ChildNode
                new_agents.append(ChildNode(new_s))
        # append any new S-nodes to the population
        self.agents.extend(new_agents)

    def run(self, t_max: float, dt: float):
        t = 0.0
        while t < t_max:
            self.step(t, dt)
            t += dt
