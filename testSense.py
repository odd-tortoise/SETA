import os

import torch

from seta import (
    Environment,
    System
)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

env = Environment(35, 1.5)

device = torch.device("cpu")
system = System(device=device)

system.initialize_system(env,4,2)

# 5) Extract PyG Data for GNN
data = system.get_graph_data()
print("Node features x:\n", data.x)
print("Edge index:\n", data.edge_index)

# 6) Plot the graph using networkx
system.plot_graph()

# 7) Query utilities
print("Number of agents:", system.num_agents())
print("Types present:", system.get_types_present())
print("Counts by type:", system.agent_counts_by_type())
print("Edge list:", system.get_edge_list())
print("All system vars:", system.get_all_system_vars())


system.reset()

env2 = Environment(15, -1.5)

system.initialize_system(env2,1,2)

# 5) Extract PyG Data for GNN
data = system.get_graph_data()
print("Node features x:\n", data.x)
print("Edge index:\n", data.edge_index)

# 6) Plot the graph using networkx
system.plot_graph()

# 7) Query utilities
print("Number of agents:", system.num_agents())
print("Types present:", system.get_types_present())
print("Counts by type:", system.agent_counts_by_type())
print("Edge list:", system.get_edge_list())
print("All system vars:", system.get_all_system_vars())

