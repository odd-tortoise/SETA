import os

import torch

from seta import (
    Environment,
    System,
    Dynamics,
    Simulator
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

env = Environment(10)

device = torch.device("cpu")
system = System(device=device)
dyn = Dynamics()

def STEM_rule(agent, system):
        """
        For a WorkerAgent:
          - increase 'workload' by 0.1 each step
          - if system.temperature > 20, increase 'age' by an extra 0.5
        """
        # agent.state is a WorkerState dataclass with fields (age, workload)
        agent.state.size += 0.1
        if system.get_system_var("temperature") > 20.0:
            agent.state.age += 0.5

def LEAF_rule(agent, system):
        """
        For a WorkerAgent:
          - increase 'workload' by 0.1 each step
          - if system.temperature > 20, increase 'age' by an extra 0.5
        """
        # agent.state is a WorkerState dataclass with fields (age, workload)
        agent.state.size += 0.1
        if system.get_system_var("temperature") > 20.0:
            agent.state.age += 0.5


dyn.register_rule("S", STEM_rule)
dyn.register_rule("L", LEAF_rule)

def spawn_node_SAM(system, prediction):
    current_W = system.types.count("W")
    delta = prediction - current_W
    if delta > 0.0:
        n_to_spawn = int(torch.ceil(torch.tensor(delta)).item())
        for _ in range(n_to_spawn):
            system.add_node_SAM()

from seta import (
     LinearThinker
)

linear_decision_ned = LinearThinker("cpu")

sim = Simulator(
     10,
     system=system,
     system_dynamic= dyn,
     decision_net=linear_decision_ned,
     act_rule=spawn_node_SAM,
     device= "cpu"
)

sim.run(env, mode = "train",verbose = 3, output="out/")

