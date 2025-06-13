
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Union
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

       
import numpy as np
from dataclasses import dataclass, asdict, fields

from .nodes import Node, NodeState
from .nodes import (
    StemNode,
    SAMNode,
    LeafNode
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


class System:
    """
    Maintains the state of a dynamic system of nodes organized as a graph,
    along with a global SystemState (environment variables). nodes can be of
    various types, each carrying a dataclass state. Provides:

      - add/remove nodes
      - update individual node state
      - extract PyG Data (x, edge_index, batch) for GNNs
      - plot the graph via networkx
      - manage global system variables via SystemState
      - utility methods (node counts, types present, etc.)

    Evolution logic (stepping ages, spawning, ODE, etc.) is handled elsewhere.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        """
        Args:
            device: torch.device for storing tensors (CPU or CUDA)
        """
        # List of node instances
        self.nodes: List[Node] = []

        # Parallel list of node types (strings), same length as self.nodes
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

    def _register_type(self, node_type: str) -> int:
        """
        Ensure node_type is registered in type_to_idx. Return its integer index.
        """
        if node_type not in self.type_to_idx:
            self.type_to_idx[node_type] = self.next_type_idx
            self.next_type_idx += 1
        return self.type_to_idx[node_type]

    # ----------------------------
    # 2.2 Add/Remove nodes
    # ----------------------------

    def add_node(
        self,
        node: Node,
        node_type: str,
        connect_to: Optional[List[int]] = None
    ) -> int:
        """
        Add a new node to the system.

        Args:
          - node      : instance of Workernode, Spawnernode, or other node subclass
          - node_type : string identifier (e.g., "W", "S", "X", etc.)
          - connect_to : optional list of existing node indices to which this new node
                         will be connected via bidirectional edges.

        Returns:
          - new_idx : integer index of the newly added node
        """
        new_idx = len(self.nodes)
        self.nodes.append(node)

        # Register type and store it
        self._register_type(node_type)
        self.types.append(node_type)

        # Create bidirectional edges if requested
        if connect_to is not None:
            for other_idx in connect_to:
                if other_idx < 0 or other_idx >= new_idx:
                    raise IndexError(f"connect_to contains invalid index: {other_idx}")
                # Add both directions
                self.edge_list.append([new_idx, other_idx])
                self.edge_list.append([other_idx, new_idx])

        return new_idx

    def remove_node(self, idx: int):
        """
        Remove the node at index idx. All edges involving idx are removed,
        and indices of nodes > idx are shifted down by 1.

        Caution: This changes indices of all subsequent nodes.
        """
        N = len(self.nodes)
        if idx < 0 or idx >= N:
            raise IndexError("node index out of range")

        # 1) Filter out edges involving idx, and shift indices > idx by -1
        new_edge_list: List[List[int]] = []
        for (u, v) in self.edge_list:
            if u == idx or v == idx:
                continue
            u_new = u - 1 if u > idx else u
            v_new = v - 1 if v > idx else v
            new_edge_list.append([u_new, v_new])
        self.edge_list = new_edge_list

        # 2) Remove node and its type
        del self.nodes[idx]
        del self.types[idx]

    # ----------------------------
    # 2.3 Access / Update node State
    # ----------------------------

    def get_node_state_vector(self, idx: int) -> torch.Tensor:
        """
        Return this node’s full state as a 1D tensor, in the dataclass field order.
        """
        return self.nodes[idx].get_state_vector()

    def set_node_state_vector(self, idx: int, vec: Union[List[float], torch.Tensor]):
        """
        Overwrite this node’s state from a vector (list or tensor),
        respecting dataclass field order.
        """
        self.nodes[idx].set_state_vector(vec)


    # ----------------------------
    # 2.4 Build PyG Data for GNN
    # ----------------------------

    def get_graph_data(self) -> Data:
        """
        Convert current system state into a torch_geometric.data.Data object,
        containing:
          - x:         (N, F) node feature matrix, where F = num_types + node_state_dim
                       node_state_dim = number of fields in the node’s state dataclass
          - edge_index:(2, E) tensor of edges
          - batch:     (N,) tensor of zeros (single graph)

        If no nodes exist, returns an "empty" Data.
        """
        N = len(self.nodes)
        if N == 0:
            # Empty graph
            x_empty = torch.zeros((0, self.next_type_idx + len(fields(nodestate))),
                                   dtype=torch.float32, device=self.device)
            edge_index_empty = torch.empty((2, 0), dtype=torch.long, device=self.device)
            batch_empty = torch.zeros((0,), dtype=torch.long, device=self.device)
            return Data(x=x_empty, edge_index=edge_index_empty, batch=batch_empty)

        # 1) Determine state-field dimension from the first node's state dataclass
        
        max_state_dim = max(len(fields(agent.state)) for agent in self.nodes)


        num_types = self.next_type_idx

        # Build node feature matrix: one-hot(type) concatenated with state vector
       
        # 2) Build feature tensor
        x = torch.zeros((N, num_types + max_state_dim), dtype=torch.float32, device=self.device)

        for i, ag in enumerate(self.nodes):
            t_idx = self.type_to_idx[self.types[i]]
            x[i, t_idx] = 1.0
            state_vec = ag.get_state_vector(pad_to=max_state_dim)
            x[i, num_types : num_types + max_state_dim] = state_vec.to(self.device)


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

    def plot_graph(self, with_labels: bool = True, node_size: int = 300, folder = None):
        """
        Visualize the current node graph using networkx and matplotlib.
        Nodes are labeled by their index and colored by node type.

        Args:
          - with_labels: if True, draw node indices:type as labels
          - node_size: size of nodes in the plot
        """
        N = len(self.nodes)
        G = nx.Graph()
        G.add_nodes_from(range(N))

        # Add edges (only one direction per undirected pair: enforce u<v)
        for (u, v) in self.edge_list:
            if u < v:
                G.add_edge(u, v)

        # Layout
        pos = nx.bfs_layout(G, start = 0, align='horizontal')

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
        plt.title("node Graph (nodes labeled as idx:type)")
        plt.axis('off')
        plt.tight_layout()
        if folder:
            plt.savefig(folder)
            plt.close()
        else:   
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

    def num_nodes(self) -> int:
        """Return total number of nodes currently in the system."""
        return len(self.nodes)

    def get_types_present(self) -> List[str]:
        """Return a list (without duplicates) of node types currently present."""
        return list(set(self.types))

    def node_counts_by_type(self) -> Dict[str, int]:
        """
        Return a dict mapping each node type to the number of nodes of that type.
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
        self.nodes: List[Node] = []

        # Parallel list of node types (strings), same length as self.nodes
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
            idx = self.add_node(StemNode(), node_type="S")
            idx_leaf = self.add_node(LeafNode(), node_type="L", connect_to=[idx])
            idx_leaf2 = self.add_node(LeafNode(), node_type="L", connect_to=[idx])
            w_indices.append(idx)
        # Add S_initial spawners, connecting first spawner to last worker if exists
        s_indices = []
        for i in range(s_count):
            # connect to last worker if i == 0 and there is at least one worker
            if i == 0 and w_count > 0:
                idx = self.add_node(SAMNode(), node_type="SAM", connect_to=[w_indices[-1]])
            else:
                idx = self.add_node(SAMNode(), node_type="SAM")
            s_indices.append(idx)

        # If more than one spawner, chain them too
        for i in range(1, len(w_indices)):
             ## add leaves
            
            self.edge_list.append([w_indices[i-1], w_indices[i]])
            self.edge_list.append([w_indices[i], w_indices[i-1]])

        # If more than one spawner, chain them too
        for i in range(1, len(s_indices)):
            self.edge_list.append([s_indices[i-1], s_indices[i]])
            self.edge_list.append([s_indices[i], s_indices[i-1]])


       



        # Set temperature
        self.read_env(env=env)

    def add_node_SAM(self) -> int:
        """
        Add a new Workernode immediately “below” the first Spawner in the chain,
        removing the existing direct connection between that spawner and its worker neighbor.

        Steps:
        1. Find the index of the first node of type "S" in self.types.
        2. Among that spawner’s neighbors, find the worker it’s connected to.
        3. Remove the edge between that worker and the spawner.
        4. Insert a new Workernode and connect it to both the old worker and the spawner.

        Returns:
        - The index of the newly added Workernode.
        Raises:
        - RuntimeError if no Spawner ("S") is found or no Worker neighbor exists.
        """
        # 1) Locate the first "S"
        try:
            first_S_idx = self.types.index("SAM")
        except ValueError:
            raise RuntimeError("Cannot add: no Spawner ('S') found in the system.")

        # 2) Find a neighboring worker of that spawner
        neighbor_worker_idx: Optional[int] = None
        for (u, v) in self.edge_list:
            if u == first_S_idx and self.types[v] == "S":
                neighbor_worker_idx = v
                break
            if v == first_S_idx and self.types[u] == "S":
                neighbor_worker_idx = u
                break

        if neighbor_worker_idx is None:
            raise RuntimeError(f"Spawner at index {first_S_idx} has no STEM neighbor.")

        # 3) Remove the edge between that worker and the spawner
        new_edge_list = []
        for (u, v) in self.edge_list:
            # Skip both directions of the old worker–spawner edge
            if (u == neighbor_worker_idx and v == first_S_idx) or (u == first_S_idx and v == neighbor_worker_idx):
                continue
            new_edge_list.append([u, v])
        self.edge_list = new_edge_list

        # 4) Add the new Workernode
        new_idx = self.add_node(StemNode(), node_type="S")

        idx_leaf = self.add_node(LeafNode(), node_type="L", connect_to=[new_idx])
        idx_leaf2 = self.add_node(LeafNode(), node_type="L", connect_to=[new_idx])

        # 5) Connect new worker to the original worker and to the first spawner
        self.edge_list.append([new_idx, neighbor_worker_idx])
        self.edge_list.append([neighbor_worker_idx, new_idx])

        self.edge_list.append([new_idx, first_S_idx])
        self.edge_list.append([first_S_idx, new_idx])

        return new_idx
    
    def _leaf_function_standard(self, angle: float, t: float) -> np.ndarray:
        """
        Standard leaf shape function.

        Args:
            angle (float): The angle for the leaf shape.
            t (float): The time parameter for the leaf shape.

        Returns:
            np.ndarray: The coordinates of the leaf shape.
        """

        def gieles(theta: float, m: int, n1: float, n2: float, n3: float, a: float, b: float) -> float:
            r = (np.abs(np.cos(m * theta / 4) / a)) ** n2 + (np.abs(np.sin(m * theta / 4) / b)) ** n3
            r = r ** (-1 / n1)
            return r

        r = gieles(angle, 2, 1, 1, 1, 2 * t, t)
        x = r * np.cos(angle) + t
        y = r * np.sin(angle)

        return np.array([x, y, 0])


    def _generate_leaf_points(self, size , angle_with_z: float = 0, angle_with_y: float = 0, n_points: int = 11) -> List[np.ndarray]:
        """
        Generate points for a single leaf.

        Args:
            state (Any): The state of the node containing required variables.
            angle_with_z (float): The angle to rotate around the z-axis.
            angle_with_y (float): The angle to rotate around the y-axis.
            n_points (int): The number of points to generate along the leaf.

        Returns:
            List[np.ndarray]: A list of points representing the leaf.
        """
        temp_points = [self._leaf_function_standard(theta, size) for theta in np.linspace(0, 2 * np.pi, n_points)]

        rot_y = np.array([
            [np.cos(angle_with_y), 0, np.sin(angle_with_y)],
            [0, 1, 0],
            [-np.sin(angle_with_y), 0, np.cos(angle_with_y)]
        ])

        rot_z = np.array([
            [np.cos(angle_with_z), -np.sin(angle_with_z), 0],
            [np.sin(angle_with_z), np.cos(angle_with_z), 0],
            [0, 0, 1]
        ])

        points = [np.dot(rot_y, point) for point in temp_points]
        points = [np.dot(rot_z, point) for point in points]
       
        return points

    def graph_to_plant(self, folder = None):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Start plotting from origin
        current_pos = np.array([0.0, 0.0, 0.0])
        current_z = 0.0


        leaf_count = 0  # used to alternate leaf placement
        

        for node in self.nodes:
            if isinstance(node, StemNode):
                length = node.state.size
                # Generate vertical line from current position
                z = np.linspace(current_z, current_z + length, 10)
                x = np.full_like(z, current_pos[0])
                y = np.full_like(z, current_pos[1])

                ax.plot(x, y, z, color='green', linewidth=2)

                # Update current position to top of the stem
                current_z += length
                current_pos = np.array([current_pos[0], current_pos[1], current_z])

            elif isinstance(node, LeafNode):
                size = node.state.size
                rachid_points = [np.array([0, 0, 0])]
                leaves_points = []
                size_pet = size
                leaves_to_plot = 5
                rachid_size = size
                y_angle = 0.7
                z_angle = 0 if leaf_count % 2 == 0 else np.pi

                while leaves_to_plot > 0:
                    #add rachid point
                    rachid_point = rachid_points[-1] + rachid_size * np.array([np.cos(y_angle),0, np.sin(y_angle)])
                    rachid_points.append(rachid_point)
                    
                    if leaves_to_plot >= 2:
                    
                        leaf_points_up = self._generate_leaf_points(size ,angle_with_z = np.pi/2, angle_with_y = y_angle)
                        leaf_points_down = self._generate_leaf_points(size ,angle_with_z = -np.pi/2, angle_with_y = y_angle)

                        petiole_up = np.array([0,size_pet,0])
                        petiole_down = np.array([0, -size_pet,0])

                        # translate the leaf points to the tip of the rachid
                        leaf_points_up = [point + rachid_point + petiole_up + current_pos for point in leaf_points_up]
                        lead_points_down = [point + rachid_point + petiole_down + current_pos for point in leaf_points_down]

                        leaves_points.append(leaf_points_up)
                        leaves_points.append(lead_points_down)

                        leaves_to_plot -= 2

                    if leaves_to_plot == 1:
                        # add the leaves on the sides
                        # add the leaf on the tip 
                        leaf_point = self._generate_leaf_points(size,angle_with_z = 0, angle_with_y= y_angle)
                        
                        
                        petiole = np.array([size_pet,0,0])
                        # translate the leaf points to the tip of the rachid
                        leaf_point = [point + rachid_point + petiole + current_pos for point in leaf_point]
                        

                        leaves_points.append(leaf_point)
                        leaves_to_plot -= 1

                    y_angle -= 0.5*y_angle

                
                z_rotation_angle = z_angle
                rot_z = np.array([
                    [np.cos(z_rotation_angle), -np.sin(z_rotation_angle), 0],
                    [np.sin(z_rotation_angle), np.cos(z_rotation_angle), 0],
                    [0, 0, 1]
                ])
            
                
                rotated_leaves = []
        
                for leaf in leaves_points:
                    leaf = [np.dot(rot_z, point) for point in leaf]
                    rotated_leaves.append(leaf)

                rotated_rachid = [np.dot(rot_z, point) for point in rachid_points]
                


                leaf_skeletons = rotated_leaves
                rachid_skeleton = rotated_rachid + current_pos
                rachid_skeleton = np.array(rachid_skeleton)
                if rachid_skeleton.size > 0:
                    ax.plot(rachid_skeleton[:, 0], rachid_skeleton[:, 1], rachid_skeleton[:, 2],
                            color="purple", label='Rachid Skeleton', linewidth=3)
                for pos,leaf in enumerate(leaf_skeletons):
                    leaf = np.array(leaf)
                    if leaf.size > 0:
                        ax.plot(leaf[:, 0], leaf[:, 1], leaf[:, 2],
                                color="orange", label='Leaf Skeleton', linewidth=4)
                        ax.plot([leaf[0, 0], leaf[-1, 0]], [leaf[0, 1], leaf[-1, 1]], [leaf[0, 2], leaf[-1, 2]], color="orange", linewidth=4)

                        if pos <= len(rachid_skeleton)-1:
                            ax.plot([leaf[5,0],rachid_skeleton[pos//2 +1 ,0]], [leaf[5,1],rachid_skeleton[pos//2 +1,1]], [leaf[5,2],rachid_skeleton[pos//2+1,2]], color = "blue", linewidth=2)

                ax.plot([leaf[5,0],rachid_skeleton[-1,0]], [leaf[5,1],rachid_skeleton[-1,1]], [leaf[5,2],rachid_skeleton[-1,2]], color = "blue", linewidth=2)


                

                leaf_count += 1

        # Set labels and show
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        size = current_z
    
        size = size+1 if size%2!=0 else size
        
        ax.set_xlim([-size // 2, size // 2])
        ax.set_ylim([-size // 2, size // 2])
        ax.set_zlim([0, size*.8])

        ax.grid(False)
        ax.axis('off')
        ax.view_init(elev=30, azim=130)


        plt.title("3D Plant Structure")
        if folder:
            plt.savefig(folder)
            plt.close()
        else:   
            plt.show()
