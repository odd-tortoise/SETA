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


for _ in range(4):
    print(system.get_agent_state_vector(_))

dyn.apply(system)

for _ in range(4):
    print(system.get_agent_state_vector(_))



