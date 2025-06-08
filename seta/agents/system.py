
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Union
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, fields

from .nodes import Agent, AgentState
from .nodes import (
    WorkerAgent,
    SpawnerAgent
)

from ..environment.environment import Environment


# ──────────────────────────────────────────────────────────────────────────────
# 2) System Class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SystemState:
    """
    Global system variables.
    """
    # Add more as needed
    temperature : float = 0.0
    humidity : float = 0.0


class System:
    """
    Maintains the state of a dynamic system of agents organized as a graph,
    along with a global SystemState (environment variables). Agents can be of
    various types, each carrying a dataclass state. Provides:

      - add/remove agents
      - update individual agent state
      - extract PyG Data (x, edge_index, batch) for GNNs
      - plot the graph via networkx
      - manage global system variables via SystemState
      - utility methods (agent counts, types present, etc.)

    Evolution logic (stepping ages, spawning, ODE, etc.) is handled elsewhere.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        """
        Args:
            device: torch.device for storing tensors (CPU or CUDA)
        """
        # List of Agent instances
        self.agents: List[Agent] = []

        # Parallel list of agent types (strings), same length as self.agents
        # Used for constructing one-hot type features in get_graph_data()
        self.types: List[str] = []
        self.type_to_idx: Dict[str, int] = {}
        self.next_type_idx = 0

        # Adjacency list: list of [source_idx, target_idx] pairs (bidirectional edges)
        self.edge_list: List[List[int]] = []

        # Global system state, using a dataclass
        self.system_state = SystemState()

        self.device = device

    # ----------------------------
    # 2.1 Type Registration
    # ----------------------------

    def _register_type(self, agent_type: str) -> int:
        """
        Ensure agent_type is registered in type_to_idx. Return its integer index.
        """
        if agent_type not in self.type_to_idx:
            self.type_to_idx[agent_type] = self.next_type_idx
            self.next_type_idx += 1
        return self.type_to_idx[agent_type]

    # ----------------------------
    # 2.2 Add/Remove Agents
    # ----------------------------

    def add_agent(
        self,
        agent: Agent,
        agent_type: str,
        connect_to: Optional[List[int]] = None
    ) -> int:
        """
        Add a new agent to the system.

        Args:
          - agent      : instance of WorkerAgent, SpawnerAgent, or other Agent subclass
          - agent_type : string identifier (e.g., "W", "S", "X", etc.)
          - connect_to : optional list of existing agent indices to which this new agent
                         will be connected via bidirectional edges.

        Returns:
          - new_idx : integer index of the newly added agent
        """
        new_idx = len(self.agents)
        self.agents.append(agent)

        # Register type and store it
        self._register_type(agent_type)
        self.types.append(agent_type)

        # Create bidirectional edges if requested
        if connect_to is not None:
            for other_idx in connect_to:
                if other_idx < 0 or other_idx >= new_idx:
                    raise IndexError(f"connect_to contains invalid index: {other_idx}")
                # Add both directions
                self.edge_list.append([new_idx, other_idx])
                self.edge_list.append([other_idx, new_idx])

        return new_idx

    def remove_agent(self, idx: int):
        """
        Remove the agent at index idx. All edges involving idx are removed,
        and indices of agents > idx are shifted down by 1.

        Caution: This changes indices of all subsequent agents.
        """
        N = len(self.agents)
        if idx < 0 or idx >= N:
            raise IndexError("Agent index out of range")

        # 1) Filter out edges involving idx, and shift indices > idx by -1
        new_edge_list: List[List[int]] = []
        for (u, v) in self.edge_list:
            if u == idx or v == idx:
                continue
            u_new = u - 1 if u > idx else u
            v_new = v - 1 if v > idx else v
            new_edge_list.append([u_new, v_new])
        self.edge_list = new_edge_list

        # 2) Remove agent and its type
        del self.agents[idx]
        del self.types[idx]

    # ----------------------------
    # 2.3 Access / Update Agent State
    # ----------------------------

    def get_agent_state_vector(self, idx: int) -> torch.Tensor:
        """
        Return this agent’s full state as a 1D tensor, in the dataclass field order.
        """
        return self.agents[idx].get_state_vector()

    def set_agent_state_vector(self, idx: int, vec: Union[List[float], torch.Tensor]):
        """
        Overwrite this agent’s state from a vector (list or tensor),
        respecting dataclass field order.
        """
        self.agents[idx].set_state_vector(vec)


    # ----------------------------
    # 2.4 Build PyG Data for GNN
    # ----------------------------

    def get_graph_data(self) -> Data:
        """
        Convert current system state into a torch_geometric.data.Data object,
        containing:
          - x:         (N, F) node feature matrix, where F = num_types + agent_state_dim
                       agent_state_dim = number of fields in the agent’s state dataclass
          - edge_index:(2, E) tensor of edges
          - batch:     (N,) tensor of zeros (single graph)

        If no agents exist, returns an "empty" Data.
        """
        N = len(self.agents)
        if N == 0:
            # Empty graph
            x_empty = torch.zeros((0, self.next_type_idx + len(fields(AgentState))),
                                   dtype=torch.float32, device=self.device)
            edge_index_empty = torch.empty((2, 0), dtype=torch.long, device=self.device)
            batch_empty = torch.zeros((0,), dtype=torch.long, device=self.device)
            return Data(x=x_empty, edge_index=edge_index_empty, batch=batch_empty)

        # 1) Determine state-field dimension from the first agent's state dataclass
        sample_state = self.agents[0].state
        state_fieldcount = len(fields(sample_state))  # e.g., 2 if WorkerState(age, workload)

        num_types = self.next_type_idx
        feature_dim = state_fieldcount

        # Build node feature matrix: one-hot(type) concatenated with state vector
        x = torch.zeros((N, num_types + feature_dim), dtype=torch.float32, device=self.device)
        for i, ag in enumerate(self.agents):
            t_idx = self.type_to_idx[self.types[i]]
            x[i, t_idx] = 1.0
            state_vec = ag.get_state_vector()  # length = feature_dim
            x[i, num_types:num_types + feature_dim] = state_vec.to(self.device)

        # 2) Build edge_index from edge_list
        if len(self.edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            ei = torch.tensor(self.edge_list, dtype=torch.long, device=self.device)  # shape (E,2)
            edge_index = ei.t().contiguous()  # shape (2, E)

        # 3) Batch: all zeros (single graph)
        batch = torch.zeros((N,), dtype=torch.long, device=self.device)

        return Data(x=x, edge_index=edge_index, batch=batch)

    # ----------------------------
    # 2.5 Graph Plotting (networkx)
    # ----------------------------

    def plot_graph(self, with_labels: bool = True, node_size: int = 300):
        """
        Visualize the current agent graph using networkx and matplotlib.
        Nodes are labeled by their index and colored by agent type.

        Args:
          - with_labels: if True, draw node indices:type as labels
          - node_size: size of nodes in the plot
        """
        N = len(self.agents)
        G = nx.Graph()
        G.add_nodes_from(range(N))

        # Add edges (only one direction per undirected pair: enforce u<v)
        for (u, v) in self.edge_list:
            if u < v:
                G.add_edge(u, v)

        # Layout
        pos = nx.spring_layout(G)

        # Color nodes by type index
        type_indices = [self.type_to_idx[self.types[i]] for i in range(N)]
        cmap = plt.cm.get_cmap('tab10', self.next_type_idx)
        node_colors = [cmap(type_indices[i]) for i in range(N)]

        plt.figure(figsize=(6, 6))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)
        nx.draw_networkx_edges(G, pos)
        if with_labels:
            labels = {i: f"{i}:{self.types[i]}" for i in range(N)}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.title("Agent Graph (nodes labeled as idx:type)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # 2.6 System Variables (SystemState)
    # ----------------------------

    def set_system_var(self, name: str, value: float):
        """
        Set a field in the global SystemState. If the field does not exist in SystemState,
        this will raise an AttributeError.
        """
        if hasattr(self.system_state, name):
            setattr(self.system_state, name, float(value))
        else:
            raise AttributeError(f"SystemState has no field '{name}'")

    def get_system_var(self, name: str) -> Optional[float]:
        """
        Return the value of a field in SystemState, or None if the field is not defined.
        """
        return getattr(self.system_state, name, None)

    def get_all_system_vars(self) -> Dict[str, float]:
        """Return a dict of all SystemState fields and values."""
        return asdict(self.system_state)

    # ----------------------------
    # 2.7 Utility Methods
    # ----------------------------

    def num_agents(self) -> int:
        """Return total number of agents currently in the system."""
        return len(self.agents)

    def get_types_present(self) -> List[str]:
        """Return a list (without duplicates) of agent types currently present."""
        return list(set(self.types))

    def agent_counts_by_type(self) -> Dict[str, int]:
        """
        Return a dict mapping each agent type to the number of agents of that type.
        e.g., {'W': 5, 'S': 1}
        """
        counts: Dict[str, int] = {}
        for t in self.types:
            counts[t] = counts.get(t, 0) + 1
        return counts

    def get_edge_list(self) -> List[List[int]]:
        """
        Return a shallow copy of the edge list: [[u0, v0], [u1, v1], ...]
        """
        return [edge[:] for edge in self.edge_list]
    
 
    

    def read_env(self, env : Environment):

        meas = env.__dict__

        for key,value in meas.items():
            self.set_system_var(key,value)


    def reset(self):
        self.agents: List[Agent] = []

        # Parallel list of agent types (strings), same length as self.agents
        # Used for constructing one-hot type features in get_graph_data()
        self.types: List[str] = []
        self.type_to_idx: Dict[str, int] = {}
        self.next_type_idx = 0

        # Adjacency list: list of [source_idx, target_idx] pairs (bidirectional edges)
        self.edge_list: List[List[int]] = []

        # Global system state, using a dataclass
        self.system_state = SystemState()
    

    def initialize_system(self,env : Environment, w_count, s_count):
        """
        Create a fresh System with W_initial workers and S_initial spawners connected
        in a simple chain: W0–W1–...–W_{W_initial-1}–S0–S1–...–S_{S_initial-1}.
        Set system_state.temperature accordingly.
        """
        self.reset()
        # Add W_initial workers
        w_indices = []
        for _ in range(w_count):
            idx = self.add_agent(WorkerAgent(), agent_type="W")
            w_indices.append(idx)
        # Add S_initial spawners, connecting first spawner to last worker if exists
        s_indices = []
        for i in range(s_count):
            # connect to last worker if i == 0 and there is at least one worker
            if i == 0 and w_count > 0:
                idx = self.add_agent(SpawnerAgent(), agent_type="S", connect_to=[w_indices[-1]])
            else:
                idx = self.add_agent(SpawnerAgent(), agent_type="S")
            s_indices.append(idx)

        # If more than one spawner, chain them too
        for i in range(1, len(w_indices)):
            self.edge_list.append([w_indices[i-1], w_indices[i]])
            self.edge_list.append([w_indices[i], w_indices[i-1]])

        # If more than one spawner, chain them too
        for i in range(1, len(s_indices)):
            self.edge_list.append([s_indices[i-1], s_indices[i]])
            self.edge_list.append([s_indices[i], s_indices[i-1]])

        # Set temperature
        self.read_env(env=env)


    def add_agent_SAM(self) -> int:
        """
        Add a new WorkerAgent immediately “below” the first Spawner in the chain,
        removing the existing direct connection between that spawner and its worker neighbor.

        Steps:
        1. Find the index of the first agent of type "S" in self.types.
        2. Among that spawner’s neighbors, find the worker it’s connected to.
        3. Remove the edge between that worker and the spawner.
        4. Insert a new WorkerAgent and connect it to both the old worker and the spawner.

        Returns:
        - The index of the newly added WorkerAgent.
        Raises:
        - RuntimeError if no Spawner ("S") is found or no Worker neighbor exists.
        """
        # 1) Locate the first "S"
        try:
            first_S_idx = self.types.index("S")
        except ValueError:
            raise RuntimeError("Cannot add: no Spawner ('S') found in the system.")

        # 2) Find a neighboring worker of that spawner
        neighbor_worker_idx: Optional[int] = None
        for (u, v) in self.edge_list:
            if u == first_S_idx and self.types[v] == "W":
                neighbor_worker_idx = v
                break
            if v == first_S_idx and self.types[u] == "W":
                neighbor_worker_idx = u
                break

        if neighbor_worker_idx is None:
            raise RuntimeError(f"Spawner at index {first_S_idx} has no Worker neighbor.")

        # 3) Remove the edge between that worker and the spawner
        new_edge_list = []
        for (u, v) in self.edge_list:
            # Skip both directions of the old worker–spawner edge
            if (u == neighbor_worker_idx and v == first_S_idx) or (u == first_S_idx and v == neighbor_worker_idx):
                continue
            new_edge_list.append([u, v])
        self.edge_list = new_edge_list

        # 4) Add the new WorkerAgent
        new_idx = self.add_agent(WorkerAgent(), agent_type="W")

        # 5) Connect new worker to the original worker and to the first spawner
        self.edge_list.append([new_idx, neighbor_worker_idx])
        self.edge_list.append([neighbor_worker_idx, new_idx])

        self.edge_list.append([new_idx, first_S_idx])
        self.edge_list.append([first_S_idx, new_idx])

        return new_idx
