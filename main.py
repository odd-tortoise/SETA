# main.py

from seta import Simulation, Environment, DAESolver, GenNode, ChildNode

from dataclasses import dataclass
import numpy as np


# -- State dataclasses (must match what your agents expect) --

@dataclass
class ANodeState:
    age: float = 0.0
    temperature: float = 25.0

@dataclass
class SNodeState:
    age: float = 0.0
    length: float = 0.1
    rank: float = 1.0
    temperature: float = 25.0

# -- (Optional) batched dynamics function for DAESolver --

def batched_growth(t, states: np.ndarray) -> np.ndarray:
    """
    states: Nx4 array columns = [age, length, rank, temp]
    returns d/dt for each column
    """
    ages = states[:, 0]
    lengths = states[:, 1]
    ranks = states[:, 2]
    temps = states[:, 3]

    # logistic growth dL/dt for each S-node
    Lmax = ranks * (1 + temps / 30.0)
    r = 0.5
    dlength = r * lengths * (1 - lengths / Lmax)

    # age derivative is 1
    dage = np.ones_like(ages)

    # rank & temp are constants → zero derivative
    drank = np.zeros_like(ranks)
    dtemp = np.zeros_like(temps)

    return np.stack([dage, dlength, drank, dtemp], axis=1)

# -- Main execution --

if __name__ == "__main__":
    # 1) Initialize environment
    env = Environment(temperature=25.0)

    # 2) Initialize dynamics solver (optional; SNode.evolve also works)
    solver = DAESolver(func=batched_growth)

    # 3) Create initial agents
    a_state = ANodeState()
    a_node = GenNode(state=a_state, thinker=None)   # no thinking logic yet

    # start with one S-node
    s_state = SNodeState()
    s_node = ChildNode(state=s_state)

    agents = [a_node, s_node]

    # 4) Instantiate simulation
    sim = Simulation(environment=env, agents=agents, dae_solver=solver)

    # 5) Run
    T_MAX = 5.0
    DT = 0.5
    sim.run(t_max=T_MAX, dt=DT)

    # 6) Inspect final states
    print("Final A-node:", a_node.state)
    for i, sn in enumerate(agents[1:], start=1):
        print(f"S-node {i}:", sn.state)
