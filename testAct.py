import os

import torch

from seta import (
    Environment,
    System,
    Dynamics
)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

env = Environment(5, 1.5)

device = torch.device("cpu")
system = System(device=device)

system.initialize_system(env,4,2)

dyn = Dynamics()

def worker_rule(agent, system):
        """
        For a WorkerAgent:
          - increase 'workload' by 0.1 each step
          - if system.temperature > 20, increase 'age' by an extra 0.5
        """
        # agent.state is a WorkerState dataclass with fields (age, workload)
        agent.state.size += 0.1
        if system.get_system_var("temperature") > 20.0:
            agent.state.age += 0.5

dyn.register_rule("W", worker_rule)


for _ in range(len(system.agents)):
    print(system.get_agent_state_vector(_))
print()

dyn.apply(system)

for _ in range(len(system.agents)):
    print(system.get_agent_state_vector(_))
print()



def spawn_node_SAM(system, prediction):
    current_W = system.types.count("W")
    delta = prediction - current_W
    if delta > 0.0:
        n_to_spawn = int(torch.ceil(torch.tensor(delta)).item())
        for _ in range(n_to_spawn):
            system.add_agent_SAM()

system.plot_graph()
spawn_node_SAM(system, 7)
system.plot_graph()
print()

for _ in range(len(system.agents)):
    print(system.get_agent_state_vector(_))
print()


dyn.apply(system)

for _ in range(len(system.agents)):
    print(system.get_agent_state_vector(_))

