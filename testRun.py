import os

import torch

from seta import (
    Environment,
    System,
    Dynamics,
    Simulator,
    MLPThinker
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# params

T_max= 100
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SYSTEM

system = System(device=device)


decision_net = MLPThinker([32,32])

dyn = Dynamics()

def STEM_rule(agent, system):
        """
        For a WorkerAgent:
          - increase 'workload' by 0.1 each step
          - if system.temperature > 20, increase 'age' by an extra 0.5
        """
        # agent.state is a WorkerState dataclass with fields (age, workload)
        agent.state.size += 0.1
        

def LEAF_rule(agent, system):
        """
        For a WorkerAgent:
          - increase 'workload' by 0.1 each step
          - if system.temperature > 20, increase 'age' by an extra 0.5
        """
        # agent.state is a WorkerState dataclass with fields (age, workload)
        agent.state.size += 0.1
        


dyn.register_rule("S", STEM_rule)
dyn.register_rule("L", LEAF_rule)


def spawn_node_SAM(system, prediction):
        current_W = system.types.count("S")
        delta = prediction - current_W
        if delta > 0.0:
            n_to_spawn = int(torch.ceil(torch.tensor(delta)).item())
            for _ in range(n_to_spawn):
                system.add_node_SAM()

sim = Simulator(
     T_max=T_max,
     system=system,
     system_dynamic= dyn,
     decision_net=decision_net,
     act_rule=spawn_node_SAM,
     device= device
    )


model_path = "decision_net.pth"
decision_net.load_state_dict(torch.load(model_path, map_location=device))

env_test = Environment(28)
sim.T_max = 100
with torch.no_grad():
  sim_stem_count = sim.run(env_test,"infer",3, output="out/") # torch tensor


import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the RBF interpolation from the file
with open('rbf_interpolation_internodes.pkl', 'rb') as f:
    rbf_loaded = pickle.load(f)


def curve_data(time_tensor: torch.Tensor, temperature: float) -> torch.Tensor:
    pred = []
    for el in time_tensor:
        pred.append(torch.tensor(rbf_loaded(np.array([el.item(),temperature]).reshape(1,-1))).float())
    return torch.cat(pred, dim=0)




# Assume time range corresponds to sim_stem_count length over T_max
time_tensor = torch.linspace(0, sim.T_max, len(sim_stem_count))

# Compute fitted (interpolated) stem count curve from RBF
fitted_curve = curve_data(time_tensor, env_test.temperature)

# Convert to numpy for plotting
time_np = time_tensor.numpy()
sim_stem_np = np.array(sim_stem_count)
fitted_np = fitted_curve.numpy()

# Plot
plt.figure(figsize=(10,6))
plt.plot(time_np, sim_stem_np, label='Simulated Stem Count', linewidth=2, marker='o')
plt.plot(time_np, fitted_np, label='RBF Interpolated Stem Count', linewidth=2, linestyle='--')

plt.xlabel('Time')
plt.ylabel('Stem Count')
plt.title('Comparison: Simulated vs RBF-Fitted Stem Growth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()